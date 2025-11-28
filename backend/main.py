# main.py

import os
import json
import asyncio
import datetime
from dotenv import load_dotenv
import httpx
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from pathlib import Path


from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager

from datatype import ImageInfo, PatientData, SubmitDiagnosisRequest, ManualDiagnosisData, UpdateSelectionRequest, AlterThresholdRequest, CustomDiseases
from patientdataio import load_batch_patient_data, create_batch_dummy_patient_data, load_patient_from_record

import logging

# 在文件开头添加日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

MAINTENANCE_CONFIG = {
    # Toggle maintenance mode without touching other runtime code paths
    "enabled": os.getenv("MAINTENANCE_MODE", "false").lower() == "true",
    "message": os.getenv("MAINTENANCE_MESSAGE", "系统维护中，请稍后再试。"),
    # Keep essential paths reachable so status checks and docs still load
    "allow_paths": {
        "/api/maintenance",
        "/api/monitor_status",
        "/docs",
        "/openapi.json",
        "/redoc",
    },
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure initial data load and start monitor
    logger.info("Application startup: loading initial data...")
    _refresh_inference_cache(force=True)
    _start_background_monitor()
    yield
    # Shutdown: stop the monitor
    logger.info("Application shutdown: stopping background monitor...")
    _stop_background_monitor()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def maintenance_guard(request: Request, call_next):
    """Return a friendly 503 when maintenance mode is enabled."""
    if MAINTENANCE_CONFIG.get("enabled"):
        path = request.url.path.rstrip("/") or "/"
        if path not in MAINTENANCE_CONFIG.get("allow_paths", set()):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "maintenance",
                    "message": MAINTENANCE_CONFIG.get("message", "Service under maintenance."),
                },
            )
    return await call_next(request)


# Get data paths from environment variables (must be defined before functions that use them)
RAW_JSON_ROOT_ENV = os.getenv("RAW_JSON_PATH", "../data")
if not os.path.isabs(RAW_JSON_ROOT_ENV):
    RAW_JSON_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), RAW_JSON_ROOT_ENV))
else:
    RAW_JSON_ROOT = RAW_JSON_ROOT_ENV

RAW_JSON_MAX_DAYS = max(1, int(os.getenv("RAW_JSON_MAX_DAYS", "21")))

RAW_JSON_MODELS_PATH_ENV = os.getenv("RAW_JSON_MODELS_PATH")
if RAW_JSON_MODELS_PATH_ENV and not os.path.isabs(RAW_JSON_MODELS_PATH_ENV):
    RAW_JSON_MODELS_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), RAW_JSON_MODELS_PATH_ENV)
    )
else:
    RAW_JSON_MODELS_PATH = RAW_JSON_MODELS_PATH_ENV

EXAMINE_RESULTS_PATH_ENV = os.getenv("EXAMINE_RESULTS_PATH", "../data/examine_results.json")
if not os.path.isabs(EXAMINE_RESULTS_PATH_ENV):
    EXAMINE_RESULTS_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), EXAMINE_RESULTS_PATH_ENV)
    )
else:
    EXAMINE_RESULTS_PATH = EXAMINE_RESULTS_PATH_ENV

INFERENCE_RESULTS_PATH = EXAMINE_RESULTS_PATH

DEFAULT_DISEASES = [
    {
        "key": "青光眼",
        "label_cn": "青光眼",
        "label_en": "Glaucoma",
        "short_name": "Glaucoma",
        "category": "glaucoma",
        "color": "text-purple-600",
        "aliases": ["青光眼"]
    },
    {
        "key": "糖网",
        "label_cn": "糖网",
        "label_en": "Diabetic Retinopathy",
        "short_name": "DR",
        "category": "retinal",
        "color": "text-red-600",
        "aliases": ["糖尿病性视网膜病变", "糖网"]
    },
    {
        "key": "AMD",
        "label_cn": "年龄相关性黄斑变性",
        "label_en": "Age-related Macular Degeneration",
        "short_name": "AMD",
        "category": "macular",
        "color": "text-orange-600",
        "aliases": ["年龄相关性黄斑变性", "AMD"]
    },
    {
        "key": "病理性近视",
        "label_cn": "病理性近视",
        "label_en": "Pathological Myopia",
        "short_name": "PM",
        "category": "macular",
        "color": "text-yellow-600",
        "aliases": ["病理性近视"]
    },
    {
        "key": "RVO",
        "label_cn": "视网膜静脉阻塞",
        "label_en": "Retinal Vein Occlusion",
        "short_name": "RVO",
        "category": "vascular",
        "color": "text-pink-600",
        "aliases": ["视网膜静脉阻塞", "视网膜静脉阻塞（RVO）", "RVO"]
    },
    {
        "key": "RAO",
        "label_cn": "视网膜动脉阻塞",
        "label_en": "Retinal Artery Occlusion",
        "short_name": "RAO",
        "category": "vascular",
        "color": "text-rose-600",
        "aliases": ["视网膜动脉阻塞", "视网膜动脉阻塞（RAO）", "RAO"]
    },
    {
        "key": "视网膜脱离",
        "label_cn": "视网膜脱离",
        "label_en": "Retinal Detachment",
        "short_name": "RD",
        "category": "retinal",
        "color": "text-indigo-600",
        "aliases": ["视网膜脱离", "视网膜脱离（RD）"]
    },
    {
        "key": "其它视网膜病",
        "label_cn": "其它视网膜病",
        "label_en": "Other Retinal Diseases",
        "short_name": "Other Retinal",
        "category": "retinal",
        "color": "text-cyan-600",
        "aliases": ["其它视网膜病", "其他视网膜病"]
    },
    {
        "key": "其它黄斑病变",
        "label_cn": "其它黄斑病变",
        "label_en": "Other Macular Diseases",
        "short_name": "Other Macular",
        "category": "macular",
        "color": "text-teal-600",
        "aliases": ["其它黄斑病变", "其他黄斑病变"]
    },
    {
        "key": "其它眼底病变",
        "label_cn": "其它眼底病变",
        "label_en": "Other Fundus Diseases",
        "short_name": "Other Fundus",
        "category": "fundus",
        "color": "text-teal-700",
        "aliases": ["其它眼底病变", "其他眼底病变"]
    },
    {
        "key": "白内障",
        "label_cn": "白内障",
        "label_en": "Cataract",
        "short_name": "Cataract",
        "category": "lens",
        "color": "text-blue-600",
        "aliases": ["白内障"]
    },
    {
        "key": "正常",
        "label_cn": "正常",
        "label_en": "Normal",
        "short_name": "Normal",
        "category": "normal",
        "color": "text-green-600",
        "aliases": ["正常", "Normal"],
        "is_normal": True
    },
]

DEFAULT_DISEASE_META = {entry["key"]: entry for entry in DEFAULT_DISEASES}

CROSS_DISEASE_ALIAS_GROUPS = [
    ["青光眼", "Glaucoma"],
    ["糖网", "糖尿病性视网膜病变", "糖尿病视网膜病变", "Diabetic Retinopathy", "DR"],
    ["AMD", "年龄相关性黄斑变性", "Age-related Macular Degeneration"],
    ["病理性近视", "高度近视", "Pathological Myopia", "Pathologic Myopia", "PM"],
    ["RVO", "视网膜静脉阻塞", "视网膜静脉阻塞（RVO）", "Retinal Vein Occlusion"],
    ["RAO", "视网膜动脉阻塞", "视网膜动脉阻塞（RAO）", "Retinal Artery Occlusion"],
    ["视网膜脱离", "视网膜脱离（RD）", "Retinal Detachment", "RD"],
    ["其它视网膜病", "其他视网膜病", "Other Retinal Diseases", "Other Retinal"],
    ["其它黄斑病变", "其他黄斑病变", "Other Macular Diseases", "Other Macular"],
    ["其它眼底病变", "其他眼底病变", "Other Fundus Diseases", "Other Fundus"],
    ["白内障", "Cataract"],
    ["正常", "Normal", "Healthy"],
]

DEFAULT_THRESHOLD_SETS = [
    {
        "id": "set_1",
        "name": "阈值套装 1",
        "description": "基准阈值（F1最优）",
        "values": {
            "青光眼": 0.20,
            "糖网": 0.28,
            "AMD": 0.25,
            "病理性近视": 0.18,
            "RVO": 0.20,
            "RAO": 0.18,
            "视网膜脱离": 0.18,
            "其它视网膜病": 0.25,
            "其它黄斑病变": 0.23,
            "其它眼底病变": 0.30,
            "白内障": 0.28,
            "正常": 0.50,
        },
    },
    {
        "id": "set_2",
        "name": "阈值套装 2",
        "description": "偏敏感阈值（参考方案）",
        "values": {
            "青光眼": 0.18,
            "糖网": 0.23,
            "AMD": 0.18,
            "病理性近视": 0.18,
            "RVO": 0.18,
            "RAO": 0.15,
            "视网膜脱离": 0.15,
            "其它视网膜病": 0.35,
            "其它黄斑病变": 0.50,
            "其它眼底病变": 0.35,
            "白内障": 0.33,
            "正常": 0.38,
        },
    },
]


def _resolve_model_root(path_value: str) -> str:
    if not path_value:
        return RAW_JSON_ROOT
    if not os.path.isabs(path_value):
        return os.path.normpath(os.path.join(os.path.dirname(__file__), path_value))
    return path_value


def _normalize_diseases(raw_diseases: Optional[List[Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Normalize disease metadata list and build alias map.
    Returns (normalized_list, alias_map).
    """
    normalized: List[Dict[str, Any]] = []
    alias_map: Dict[str, str] = {}
    if not raw_diseases:
        raw_diseases = DEFAULT_DISEASES

    for item in raw_diseases:
        if isinstance(item, str):
            entry = {"key": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            continue

        key = entry.get("key") or entry.get("name")
        if not key:
            continue
        entry["key"] = key

        defaults = DEFAULT_DISEASE_META.get(key, {})
        entry.setdefault("label_cn", defaults.get("label_cn", key))
        entry.setdefault("label_en", defaults.get("label_en", entry["label_cn"]))
        entry.setdefault("short_name", defaults.get("short_name", entry["label_en"]))
        entry.setdefault("category", defaults.get("category", "other"))
        entry.setdefault("color", defaults.get("color", "text-gray-600"))

        aliases = []
        if defaults.get("aliases"):
            aliases.extend(defaults["aliases"])
        if entry.get("aliases"):
            if isinstance(entry["aliases"], list):
                aliases.extend(entry["aliases"])
            else:
                aliases.append(entry["aliases"])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [a for a in aliases if isinstance(a, str)]

        # Expand aliases with cross-language/common mappings so different models align.
        for group in CROSS_DISEASE_ALIAS_GROUPS:
            if key in group or any(a in group for a in aliases):
                aliases.extend(group)

        # Ensure key is included and remove duplicates while preserving order.
        seen = set()
        normalized_aliases = []
        for alias in aliases + [key]:
            if not isinstance(alias, str):
                continue
            if alias in seen:
                continue
            seen.add(alias)
            normalized_aliases.append(alias)
        entry["aliases"] = normalized_aliases

        if "is_normal" in entry:
            entry["is_normal"] = bool(entry["is_normal"])
        else:
            entry["is_normal"] = bool(defaults.get("is_normal", False))

        parent_key = entry.get("parent_key")
        if parent_key is None and defaults.get("parent_key"):
            parent_key = defaults.get("parent_key")
        if parent_key:
            entry["parent_key"] = parent_key
        else:
            entry.pop("parent_key", None)

        normalized.append(entry)

        for alias in aliases:
            alias_map[alias] = key
            alias_map[alias.lower()] = key

    return normalized, alias_map


def _normalize_threshold_sets(raw_sets: Optional[List[Dict[str, Any]]], disease_keys: List[str]) -> List[Dict[str, Any]]:
    if not raw_sets:
        raw_sets = DEFAULT_THRESHOLD_SETS

    normalized_sets: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_sets):
        if not isinstance(item, dict):
            continue
        set_id = item.get("id") or f"set_{idx + 1}"
        name = item.get("name") or f"阈值套装 {idx + 1}"
        description = item.get("description", "")
        values = item.get("values") or {}
        normalized_values: Dict[str, float] = {}
        for disease_key in disease_keys:
            if disease_key in values:
                normalized_values[disease_key] = float(values[disease_key])
            else:
                normalized_values[disease_key] = 0.5
        normalized_sets.append({
            "id": set_id,
            "name": name,
            "description": description,
            "values": normalized_values
        })
    return normalized_sets


def _load_inference_models_config() -> Dict[str, Dict[str, Any]]:
    raw_models_str = os.getenv("RAW_JSON_MODELS", "").strip()
    if not raw_models_str and RAW_JSON_MODELS_PATH:
        try:
            with open(RAW_JSON_MODELS_PATH, "r", encoding="utf-8") as f:
                raw_models_str = f.read()
        except FileNotFoundError:
            logger.error("RAW_JSON_MODELS_PATH %s not found", RAW_JSON_MODELS_PATH)
        except Exception as exc:
            logger.error("Failed to read RAW_JSON_MODELS_PATH %s: %s", RAW_JSON_MODELS_PATH, exc)

    parsed_models: Any = []
    if raw_models_str:
        try:
            parsed_models = json.loads(raw_models_str)
        except json.JSONDecodeError:
            logger.error("Model configuration JSON is invalid; falling back to default dataset")
            parsed_models = []

    if isinstance(parsed_models, dict):
        parsed_models = [parsed_models]

    if not parsed_models:
        parsed_models = [
            {
                "id": "default",
                "name": "默认模型",
                "priority": 100,
                "root": RAW_JSON_ROOT_ENV,
                "diseases": DEFAULT_DISEASES,
                "threshold_sets": DEFAULT_THRESHOLD_SETS,
                "default_threshold_set_id": DEFAULT_THRESHOLD_SETS[0]["id"],
            }
        ]

    configs: Dict[str, Dict[str, Any]] = {}
    for idx, raw_entry in enumerate(parsed_models):
        if not isinstance(raw_entry, dict):
            continue
        model_id = raw_entry.get("id") or f"model_{idx + 1}"
        if model_id in configs:
            logger.warning("Duplicate model id %s found in RAW_JSON_MODELS; keeping first occurrence", model_id)
            continue

        diseases, alias_map = _normalize_diseases(raw_entry.get("diseases"))
        disease_keys = [d["key"] for d in diseases]
        threshold_sets = _normalize_threshold_sets(raw_entry.get("threshold_sets"), disease_keys)
        if not threshold_sets:
            threshold_sets = _normalize_threshold_sets(DEFAULT_THRESHOLD_SETS, disease_keys)

        default_set_id = raw_entry.get("default_threshold_set_id") or threshold_sets[0]["id"]
        threshold_indices = {s["id"]: idx for idx, s in enumerate(threshold_sets)}

        cfg = {
            "id": model_id,
            "name": raw_entry.get("name") or model_id,
            "priority": int(raw_entry.get("priority", 0)),
            "root": raw_entry.get("root") or raw_entry.get("path") or RAW_JSON_ROOT_ENV,
            "abs_root": None,  # resolved below
            "diseases": diseases,
            "disease_alias_map": alias_map,
            "threshold_sets": threshold_sets,
            "threshold_set_indices": threshold_indices,
            "default_threshold_set_id": default_set_id if default_set_id in threshold_indices else threshold_sets[0]["id"],
        }
        cfg["abs_root"] = _resolve_model_root(cfg["root"])
        configs[model_id] = cfg

    if not configs:
        raise RuntimeError("No valid inference model configuration detected")

    return configs


MODEL_CONFIGS = _load_inference_models_config()
ORDERED_MODEL_IDS = sorted(
    MODEL_CONFIGS.keys(),
    key=lambda mid: (-MODEL_CONFIGS[mid]["priority"], MODEL_CONFIGS[mid]["name"])
)
DEFAULT_MODEL_ID = ORDERED_MODEL_IDS[0]

# --- Data Storage (In-memory cache) ---
patients_data_cache: Dict[str, PatientData] = {}
# Storage for manual diagnosis data (separate from AI predictions)
manual_diagnosis_storage: Dict[str, ManualDiagnosisData] = {}
# Cache of raw per-image probabilities to avoid disk I/O on reselection
raw_probs_cache: Dict[str, Dict[str, Any]] = {}
manual_diagnosis_file_lock = threading.Lock()


def _resolve_model_id(model_id: Optional[str]) -> str:
    if model_id and model_id in MODEL_CONFIGS:
        return model_id
    if model_id and model_id not in MODEL_CONFIGS:
        logger.warning("Requested model_id %s not found; falling back to default %s", model_id, DEFAULT_MODEL_ID)
    return DEFAULT_MODEL_ID


def _get_model_config(model_id: Optional[str] = None) -> Dict[str, Any]:
    resolved = _resolve_model_id(model_id)
    return MODEL_CONFIGS[resolved]


def _make_patient_cache_key(model_id: str, patient_id: str, exam_date: Optional[str] = None) -> str:
    suffix = exam_date or "latest"
    return f"{model_id}::{patient_id}::{suffix}"


def _make_raw_cache_key(model_id: str, patient_id: str, exam_date: Optional[str] = None) -> str:
    return _make_patient_cache_key(model_id, patient_id, exam_date)


def _get_cached_patient(patient_id: str, model_id: Optional[str] = None, exam_date: Optional[str] = None) -> Tuple[Optional[str], Optional[PatientData]]:
    """
    Retrieve a patient from the cache, returning (model_id, patient).
    If model_id is None, search across all models.
    """
    candidate_models = (
        [_resolve_model_id(model_id)]
        if model_id
        else ORDERED_MODEL_IDS
    )

    for mid in candidate_models:
        keys_to_check = []
        if exam_date:
            keys_to_check.append(_make_patient_cache_key(mid, patient_id, exam_date))
        keys_to_check.append(_make_patient_cache_key(mid, patient_id, None))
        for key in keys_to_check:
            patient = patients_data_cache.get(key)
            if patient:
                return mid, patient
    return None, None

def _load_manual_diagnosis_store() -> Dict[str, Any]:
    """
    Read persisted manual diagnosis information.
    
    Returns dict with structure:
    - New format: {patient_id: {exam_date: {manual_diagnosis_data}, ...}, ...}
    - Old format (auto-converted): {patient_id: {manual_diagnosis_data}} -> {patient_id: {"20251014": {manual_diagnosis_data}}}
    """
    if not os.path.exists(EXAMINE_RESULTS_PATH):
        return {}
    try:
        with open(EXAMINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Auto-detect and convert old format to new format
            converted_data = {}
            for patient_id, content in data.items():
                if isinstance(content, dict):
                    # Check if it's old format (has manual_diagnosis key directly)
                    if "manual_diagnosis" in content or "custom_diseases" in content or "diagnosis_notes" in content:
                        # Old format - convert to new format with default date
                        logger.info(f"Converting old format manual diagnosis for patient {patient_id} to new format")
                        converted_data[patient_id] = {
                            "20251014": content  # Use today's date as default for old records
                        }
                    else:
                        # Already new format (exam_date -> diagnosis data)
                        converted_data[patient_id] = content
                else:
                    logger.warning(f"Skipping invalid entry for patient {patient_id}")
            return converted_data
        logger.warning("Manual diagnosis file %s did not contain a dict; resetting", EXAMINE_RESULTS_PATH)
    except json.JSONDecodeError:
        logger.error("Manual diagnosis file %s is not valid JSON; ignoring", EXAMINE_RESULTS_PATH)
    except Exception as exc:
        logger.error("Failed to load manual diagnosis file %s: %s", EXAMINE_RESULTS_PATH, exc)
    return {}


def _save_manual_diagnosis_store(serializable: Dict[str, Any]) -> None:
    """Persist manual diagnosis storage to JSON file."""
    os.makedirs(os.path.dirname(EXAMINE_RESULTS_PATH), exist_ok=True)
    tmp_path = EXAMINE_RESULTS_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, EXAMINE_RESULTS_PATH)
    except Exception as exc:
        logger.error("Failed to save manual diagnosis data to %s: %s", EXAMINE_RESULTS_PATH, exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _preload_manual_diagnoses() -> None:
    """
    Populate in-memory manual diagnosis cache from persisted JSON file.
    Cache key format: "patient_id" or "patient_id_exam_date" for specific exams.
    """
    try:
        persisted = _load_manual_diagnosis_store()
    except Exception as exc:
        logger.error("Unable to preload manual diagnoses: %s", exc)
        persisted = {}

    if not isinstance(persisted, dict):
        logger.warning("Preloaded manual diagnosis data is not a dict; skipping preload")
        return

    for patient_id, exam_data in persisted.items():
        if not isinstance(exam_data, dict):
            continue
        
        # exam_data is now {exam_date: diagnosis_data, ...}
        for exam_date, payload in exam_data.items():
            if not isinstance(payload, dict):
                continue
            
            # Create cache key: patient_id_exam_date
            cache_key = f"{patient_id}_{exam_date}"
            
            try:
                stored = ManualDiagnosisData(**payload)
                if getattr(stored, "manual_descriptions", None) is None:
                    stored.manual_descriptions = {"left_eye": "", "right_eye": ""}
                manual_diagnosis_storage[cache_key] = stored
                logger.debug(f"Loaded manual diagnosis for {cache_key}")
            except Exception:
                try:
                    from datatype import CustomDiseases
                    stored = ManualDiagnosisData(
                        manual_diagnosis=payload.get("manual_diagnosis", {}),
                        custom_diseases=payload.get("custom_diseases") or CustomDiseases(),
                        diagnosis_notes=payload.get("diagnosis_notes", ""),
                        doctor_id=payload.get("doctor_id"),
                        manual_descriptions=payload.get("manual_descriptions")
                    )
                    if getattr(stored, "manual_descriptions", None) is None:
                        stored.manual_descriptions = {"left_eye": "", "right_eye": ""}
                    manual_diagnosis_storage[cache_key] = stored
                    logger.debug(f"Loaded manual diagnosis for {cache_key} (with fallback)")
                except Exception as inner_exc:
                    logger.error("Failed to deserialize manual diagnosis for %s: %s", cache_key, inner_exc)


_preload_manual_diagnoses()

# Limit the amount of conversation history we replay to the model each turn
MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("LLM_MAX_HISTORY_MESSAGES", "20"))

# Ollama connection resilience tuning
OLLAMA_MAX_RETRIES = max(1, int(os.getenv("LLM_MAX_RETRIES", "2")))
OLLAMA_RETRY_DELAY_SECONDS = max(0.0, float(os.getenv("LLM_RETRY_DELAY_SECONDS", "1.0")))
OLLAMA_KEEP_ALIVE = os.getenv("LLM_KEEP_ALIVE", "30m")

# Auto-reload configuration
AUTO_RELOAD_ENABLED = os.getenv("AUTO_RELOAD_DATA", "true").lower() in ("true", "1", "yes")
AUTO_RELOAD_INTERVAL_SECONDS = max(5, int(os.getenv("AUTO_RELOAD_INTERVAL", "30")))

# Background data monitoring
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop_event = threading.Event()


def _background_data_monitor():
    """Background thread that periodically checks for new/modified inference data."""
    logger.info(
        "Starting background data monitor (interval: %s seconds, enabled: %s)",
        AUTO_RELOAD_INTERVAL_SECONDS,
        AUTO_RELOAD_ENABLED
    )
    
    while not _monitor_stop_event.is_set():
        if AUTO_RELOAD_ENABLED:
            try:
                _refresh_inference_cache(force=False)
            except Exception as exc:
                logger.error("Error in background data monitor: %s", exc)
        
        # Wait for the interval or until stop event is set
        _monitor_stop_event.wait(timeout=AUTO_RELOAD_INTERVAL_SECONDS)
    
    logger.info("Background data monitor stopped")


def _start_background_monitor():
    """Start the background monitoring thread."""
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        logger.warning("Background monitor already running")
        return
    
    _monitor_stop_event.clear()
    _monitor_thread = threading.Thread(target=_background_data_monitor, daemon=True, name="DataMonitor")
    _monitor_thread.start()
    logger.info("Background data monitor thread started")


def _stop_background_monitor():
    """Stop the background monitoring thread."""
    global _monitor_thread
    if _monitor_thread is None or not _monitor_thread.is_alive():
        return
    
    logger.info("Stopping background data monitor...")
    _monitor_stop_event.set()
    _monitor_thread.join(timeout=5)
    _monitor_thread = None


# Background monitor will be started by FastAPI lifespan event

# --- Inference data aggregation (latest dated folders) ---
_inference_cache_lock = threading.Lock()
# Changed to support multiple models -> each model holds Dict[patient_id, List[instance]]
_inference_patient_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_inference_loaded_files: Dict[str, List[str]] = {}
_inference_file_mtimes: Dict[str, Dict[str, float]] = {}


def _resolve_inference_sources(model_cfg: Dict[str, Any]) -> List[tuple]:
    """Return a list of (json_path, base_dir, label) sorted by newest first for the given model."""
    root_dir = model_cfg.get("abs_root") or RAW_JSON_ROOT
    if os.path.isfile(root_dir):
        return [(root_dir, os.path.dirname(root_dir), os.path.basename(root_dir))]

    if not os.path.isdir(root_dir):
        logger.error("RAW JSON root does not exist or is not accessible: %s", root_dir)
        return []

    entries: List[tuple] = []
    try:
        for name in os.listdir(root_dir):
            full_dir = os.path.join(root_dir, name)
            if not os.path.isdir(full_dir):
                continue
            label = name.strip()
            if len(label) == 8 and label.isdigit():
                candidate = os.path.join(full_dir, "inference_results.json")
                if os.path.isfile(candidate):
                    entries.append((candidate, full_dir, label))
    except Exception as exc:
        logger.error("Failed to enumerate inference directories under %s: %s", root_dir, exc)
        return []

    if not entries:
        # Fallback: look for inference_results.json directly under root for backwards compatibility
        fallback = os.path.join(root_dir, "inference_results.json")
        if os.path.isfile(fallback):
            return [(fallback, root_dir, os.path.basename(root_dir))]
        logger.warning("No dated inference folders found under %s", root_dir)
        return []

    # Sort by label descending (newest date first)
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries[:RAW_JSON_MAX_DAYS]


def _refresh_inference_cache(force: bool = False, model_id: Optional[str] = None) -> None:
    """
    Refresh cached inference data for one or all models.
    """
    target_model_ids = (
        [_resolve_model_id(model_id)]
        if model_id
        else ORDERED_MODEL_IDS
    )

    for mid in target_model_ids:
        model_cfg = MODEL_CONFIGS[mid]
        sources = _resolve_inference_sources(model_cfg)
        file_keys = [path for path, _, _ in sources]

        with _inference_cache_lock:
            patient_records = _inference_patient_records.setdefault(mid, {})
            loaded_files = _inference_loaded_files.setdefault(mid, [])
            file_mtimes = _inference_file_mtimes.setdefault(mid, {})

            needs_reload = force or file_keys != loaded_files

            if not needs_reload:
                for path in file_keys:
                    try:
                        current_mtime = os.path.getmtime(path)
                        if file_mtimes.get(path) != current_mtime:
                            needs_reload = True
                            logger.info("Detected modification in %s for model %s", path, mid)
                            break
                    except Exception:
                        needs_reload = True
                        break

            if not needs_reload and patient_records:
                continue

            combined: Dict[str, List[Dict[str, Any]]] = {}

            for path, base_dir, label in sources:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as exc:
                    logger.error("Failed to load inference file %s for model %s: %s", path, mid, exc)
                    continue

                if not isinstance(data, dict):
                    logger.warning("Inference file %s did not contain a JSON object", path)
                    continue

                for pid, payload in data.items():
                    if not isinstance(payload, dict):
                        continue

                    instance = {
                        "data": payload,
                        "source_date": label,
                        "source_file": path,
                        "base_dir": base_dir,
                        "model_id": mid,
                    }

                    combined.setdefault(pid, []).append(instance)

            for pid in combined:
                combined[pid].sort(key=lambda x: x["source_date"], reverse=True)

            _inference_patient_records[mid] = combined
            _inference_loaded_files[mid] = file_keys
            mtimes: Dict[str, float] = {}
            for path in file_keys:
                try:
                    mtimes[path] = os.path.getmtime(path)
                except Exception:
                    pass
            _inference_file_mtimes[mid] = mtimes

            total_instances = sum(len(instances) for instances in combined.values())
            logger.info(
                "Model %s loaded %s inference file(s), %s patient(s), %s exam instance(s)",
                model_cfg["name"],
                len(file_keys),
                len(combined),
                total_instances,
            )


def get_inference_patient_ids(model_id: Optional[str] = None) -> List[str]:
    """Return list of all patient IDs for a model."""
    resolved = _resolve_model_id(model_id)
    _refresh_inference_cache(model_id=resolved)
    with _inference_cache_lock:
        return list(_inference_patient_records.get(resolved, {}).keys())


def get_inference_record(patient_id: str, exam_date: Optional[str] = None, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get inference record for a patient.
    If exam_date is provided, return that specific instance; otherwise return latest.
    Returns the raw data payload (not the metadata wrapper).
    """
    resolved = _resolve_model_id(model_id)
    _refresh_inference_cache(model_id=resolved)
    with _inference_cache_lock:
        model_records = _inference_patient_records.get(resolved, {})
        instances = model_records.get(patient_id)
        if not instances:
            return None
        
        # If specific exam_date requested, find matching instance
        if exam_date:
            for instance in instances:
                if instance["source_date"] == exam_date:
                    return dict(instance["data"])
            return None  # No match found
        
        # Return latest (first in sorted list)
        return dict(instances[0]["data"])


def get_inference_base_dir(patient_id: str, exam_date: Optional[str] = None, model_id: Optional[str] = None) -> Optional[str]:
    """
    Get base directory for a patient's exam instance.
    If exam_date is provided, return that specific instance's base_dir; otherwise return latest.
    """
    resolved = _resolve_model_id(model_id)
    _refresh_inference_cache(model_id=resolved)
    with _inference_cache_lock:
        model_records = _inference_patient_records.get(resolved, {})
        instances = model_records.get(patient_id)
        if not instances:
            return None
        
        # If specific exam_date requested, find matching instance
        if exam_date:
            for instance in instances:
                if instance["source_date"] == exam_date:
                    return instance["base_dir"]
            return None
        
        # Return latest
        return instances[0]["base_dir"]


def get_inference_instances(patient_id: str, model_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Get list of all exam instances for a patient.
    Returns list of metadata dicts with keys: source_date, source_file
    """
    resolved = _resolve_model_id(model_id)
    _refresh_inference_cache(model_id=resolved)
    with _inference_cache_lock:
        model_records = _inference_patient_records.get(resolved, {})
        instances = model_records.get(patient_id)
        if not instances:
            return []
        
        # Return lightweight metadata (don't include full data payload)
        return [
            {
                "exam_date": inst["source_date"],
                "source_file": inst["source_file"]
            }
            for inst in instances
        ]


def get_full_inference_map(model_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get all patient records (latest instance for each patient).
    Maintained for backwards compatibility.
    """
    resolved = _resolve_model_id(model_id)
    _refresh_inference_cache(model_id=resolved)
    with _inference_cache_lock:
        model_records = _inference_patient_records.get(resolved, {})
        return {
            pid: dict(instances[0]["data"])
            for pid, instances in model_records.items()
            if instances
        }

# Path to questionnaire data from the other project
CONSULTATION_DATA_PATH_ENV = os.getenv("CONSULTATION_DATA_PATH")
if CONSULTATION_DATA_PATH_ENV:
    # If path is relative, make it relative to the backend directory
    if not os.path.isabs(CONSULTATION_DATA_PATH_ENV):
        CONSULTATION_DATA_PATH = os.path.normpath(
            os.path.join(os.path.dirname(__file__), CONSULTATION_DATA_PATH_ENV)
        )
    else:
        CONSULTATION_DATA_PATH = CONSULTATION_DATA_PATH_ENV
else:
    # Default path
    CONSULTATION_DATA_PATH = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../eye_ai_consultation/data/questionnaire_data.json")
    )

# Lock for thread-safe access to cached data
patient_data_lock = threading.Lock()
consultation_data_lock = threading.Lock()

# --- New helpers for RAW v2 structure ---
def _extract_image_type(img: Dict[str, Any]) -> Optional[str]:
    """
    Map RAW v2 eye_classification.class to internal types:
      Left fundus -> 左眼CFP
      Right fundus -> 右眼CFP
      Left outer eye -> 左眼外眼照
      Right outer eye -> 右眼外眼照
    """
    try:
        cls = str(img.get("eye_classification", {}).get("class", "")).strip().lower()
    except Exception:
        cls = ""
    if cls == "left fundus":
        return "左眼CFP"
    if cls == "right fundus":
        return "右眼CFP"
    if cls == "left outer eye":
        return "左眼外眼照"
    if cls == "right outer eye":
        return "右眼外眼照"
    return None

def _extract_disease_probs(img: Dict[str, Any]) -> Dict[str, float]:
    """
    Use fundus_classification.diseases as disease probs if present.
    If missing or invalid, return {}.
    """
    try:
        diseases = img.get("fundus_classification", {}).get("diseases", {})
        if isinstance(diseases, dict):
            # ensure float
            return {str(k): float(v) for k, v in diseases.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}

def _parse_ts_from_imgpath(p: str) -> int:
    """
    Keep the original timestamp/digits extraction to sort 'latest' images.
    """
    try:
        base = os.path.basename(p or "")
        name, _ = os.path.splitext(base)
        parts = (name or "").split('_')
        cand = parts[-1] if parts else name
        digits = ''.join(ch for ch in (cand or "") if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0

def _warm_raw_cache_from_raw_json(model_id: str, ris_exam_id: str, exam_date: Optional[str] = None) -> None:
    """Build raw_probs_cache[...] from aggregated inference records."""
    record = get_inference_record(ris_exam_id, exam_date, model_id=model_id)
    if record is None:
        raise KeyError(f"Patient {ris_exam_id} not found in inference cache")

    images = record.get("images", [])
    if not isinstance(images, list):
        images = []

    raw_by_type: Dict[str, list] = {}
    raw_by_id: Dict[str, Dict] = {}

    for img in images:
        if not isinstance(img, dict):
            continue
        img_type = _extract_image_type(img)
        if not img_type:
            continue

        probs = _extract_disease_probs(img)

        # cache by id - include both legacy (no exam_date) and composite keys
        base_img_id = f"img_{ris_exam_id}_{img.get('img_path', '')}"
        raw_by_id[base_img_id] = probs
        if exam_date:
            composite_img_id = f"img_{ris_exam_id}_{exam_date}_{img.get('img_path', '')}"
            raw_by_id[composite_img_id] = probs

        # push into type list; augment with 'probs' for fallback users
        img_copy = dict(img)
        img_copy["probs"] = probs
        raw_by_type.setdefault(img_type, []).append(img_copy)

    # sort lists latest-first by parsed digits from img_path
    for k in list(raw_by_type.keys()):
        raw_by_type[k] = sorted(
            raw_by_type[k],
            key=lambda im: _parse_ts_from_imgpath(im.get("img_path", "")),
            reverse=True
        )

    # Use composite key for cache if exam_date specified
    cache_key = _make_raw_cache_key(model_id, ris_exam_id, exam_date)
    raw_probs_cache[cache_key] = {"by_type": raw_by_type, "by_id": raw_by_id}

# No preloading - data will be loaded on-demand

# --- API Endpoints ---
@app.get("/api/patients/{ris_exam_id}")
async def get_patient_by_id(ris_exam_id: str, exam_date: Optional[str] = None, model_id: Optional[str] = None):
    """
    Returns the data for a specific patient by ris_exam_id (patient_id).
    Optional exam_date parameter to load a specific exam instance (format: YYYYMMDD).
    If not provided, returns the latest exam instance.
    """
    import time
    start_time = time.time()
    
    resolved_model_id = _resolve_model_id(model_id)
    model_cfg = _get_model_config(resolved_model_id)
    cache_key = _make_patient_cache_key(resolved_model_id, ris_exam_id, exam_date)
    
    with patient_data_lock:
        # Check cache first
        if cache_key in patients_data_cache:
            print(f"Serving cached data for patient: {cache_key} (took {time.time() - start_time:.2f}s)")
            # Warm raw prob cache if missing
            raw_cache_key = _make_raw_cache_key(resolved_model_id, ris_exam_id, exam_date)
            if raw_cache_key not in raw_probs_cache:
                try:
                    _warm_raw_cache_from_raw_json(resolved_model_id, ris_exam_id, exam_date)
                except Exception:
                    pass
            return patients_data_cache[cache_key]
        
        # Load from data source
        try:
            print(f"Loading patient {cache_key} from data source...")
            record = get_inference_record(ris_exam_id, exam_date, model_id=resolved_model_id)
            if record is None:
                raise KeyError
            base_dir = get_inference_base_dir(ris_exam_id, exam_date, model_id=resolved_model_id) or model_cfg.get("abs_root") or RAW_JSON_ROOT
            patient_data = load_patient_from_record(ris_exam_id, record, base_dir, model_cfg, exam_date=exam_date)
            # Cache the loaded data with composite key
            patients_data_cache[cache_key] = patient_data
            # Build raw prob cache for this patient (warm for reselection)
            try:
                _warm_raw_cache_from_raw_json(resolved_model_id, ris_exam_id, exam_date)
            except Exception:
                pass
            elapsed = time.time() - start_time
            print(f"Loaded and cached data for patient: {cache_key} (took {elapsed:.2f}s)")
            return patient_data
        except KeyError:
            print(f"Patient {cache_key} not found (took {time.time() - start_time:.2f}s)")
            raise HTTPException(status_code=404, detail=f"Patient {ris_exam_id} (exam_date={exam_date}) not found")
        except Exception as e:
            print(f"Error loading patient {cache_key}: {e} (took {time.time() - start_time:.2f}s)")
            raise HTTPException(status_code=500, detail="Failed to load patient data")

@app.get("/api/patients")
async def get_available_patient_ids(model_id: Optional[str] = None):
    """Returns a list of available patient IDs from the data source."""
    try:
        ids = get_inference_patient_ids(model_id=model_id)
        print(f"Found {len(ids)} available patients")
        return {"patient_ids": ids}
    except Exception as e:
        print(f"Error loading patient IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to load patient IDs")

@app.get("/api/patients/{ris_exam_id}/instances")
async def get_patient_exam_instances(ris_exam_id: str, model_id: Optional[str] = None):
    """
    Returns all available exam instances for a specific patient.
    Returns list of exam dates sorted by newest first.
    """
    try:
        instances = get_inference_instances(ris_exam_id, model_id=model_id)
        if not instances:
            raise HTTPException(status_code=404, detail=f"Patient {ris_exam_id} not found")
        return {"patient_id": ris_exam_id, "instances": instances}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading exam instances for {ris_exam_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load exam instances")


@app.get("/api/models")
async def list_inference_models():
    """Expose available inference models and metadata to the frontend."""
    models_payload = []
    for mid in ORDERED_MODEL_IDS:
        cfg = MODEL_CONFIGS[mid]
        models_payload.append({
            "id": mid,
            "name": cfg["name"],
            "priority": cfg["priority"],
            "root": cfg["root"],
            "diseases": cfg["diseases"],
            "threshold_sets": cfg["threshold_sets"],
            "default_threshold_set_id": cfg["default_threshold_set_id"],
        })
    return {
        "default_model_id": DEFAULT_MODEL_ID,
        "models": models_payload,
    }


@app.get("/api/maintenance")
async def get_maintenance_status():
    """Expose maintenance toggle to the frontend."""
    return {
        "enabled": bool(MAINTENANCE_CONFIG.get("enabled")),
        "message": MAINTENANCE_CONFIG.get("message", "系统维护中，请稍后再试。"),
    }


@app.post("/api/reload_data")
async def reload_data_manually(model_id: Optional[str] = None):
    """Manually trigger a reload of inference data from disk."""
    try:
        logger.info("Manual data reload requested")
        resolved = _resolve_model_id(model_id)
        _refresh_inference_cache(force=True, model_id=resolved)
        with _inference_cache_lock:
            model_records = _inference_patient_records.get(resolved, {})
            total_patients = len(model_records)
            total_instances = sum(len(instances) for instances in model_records.values())
        return {
            "status": "success",
            "message": "Data reloaded successfully",
            "patients_count": total_patients,
            "total_instances": total_instances
        }
    except Exception as e:
        logger.error(f"Error during manual reload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload data: {str(e)}")

@app.get("/api/monitor_status")
async def get_monitor_status(model_id: Optional[str] = None):
    """Get status of the background data monitor."""
    global _monitor_thread
    is_running = _monitor_thread is not None and _monitor_thread.is_alive()

    resolved_ids = (
        [_resolve_model_id(model_id)]
        if model_id
        else ORDERED_MODEL_IDS
    )

    with _inference_cache_lock:
        models_status = []
        total_patients = 0
        total_instances = 0
        total_files = 0
        for mid in resolved_ids:
            records = _inference_patient_records.get(mid, {})
            file_list = _inference_loaded_files.get(mid, [])
            instance_count = sum(len(instances) for instances in records.values())
            models_status.append({
                "model_id": mid,
                "model_name": MODEL_CONFIGS[mid]["name"],
                "patients_count": len(records),
                "total_instances": instance_count,
                "loaded_files": len(file_list),
                "loaded_file_paths": file_list,
            })
            total_patients += len(records)
            total_instances += instance_count
            total_files += len(file_list)

    return {
        "monitor_enabled": AUTO_RELOAD_ENABLED,
        "monitor_running": is_running,
        "reload_interval_seconds": AUTO_RELOAD_INTERVAL_SECONDS,
        "patients_count": total_patients,
        "total_instances": total_instances,
        "loaded_files": total_files,
        "models": models_status,
    }

@app.post("/api/submit_diagnosis")
async def submit_diagnosis(request: SubmitDiagnosisRequest):
    """
    Receives manual diagnosis data and image info updates from the frontend and stores them.
    Now supports exam_date to handle multiple exams per patient.
    """
    # Determine exam_date - use provided or get from patient cache
    exam_date = request.exam_date
    resolved_model_id = _resolve_model_id(request.model_id)
    cache_key = _make_patient_cache_key(resolved_model_id, request.patient_id, exam_date)
    
    with patient_data_lock:
        print(f"Received diagnosis submission for patient: {request.patient_id}, exam_date: {exam_date}")
        
        # Check if patient exists in cache
        patient = patients_data_cache.get(cache_key)
        if not patient:
            # Try without exam_date
            fallback_key = _make_patient_cache_key(resolved_model_id, request.patient_id, None)
            patient = patients_data_cache.get(fallback_key)
            if not patient:
                raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found in cache. Load patient data first.")
            # If found without exam_date, try to get exam_date from patient
            if not exam_date and hasattr(patient, 'examine_time'):
                # Try to extract date from examine_time
                try:
                    import re
                    match = re.search(r'(\d{8})', patient.examine_time or "")
                    if match:
                        exam_date = match.group(1)
                        logger.info(f"Extracted exam_date {exam_date} from examine_time")
                except Exception:
                    pass
            
            # If still no exam_date, check inference records to get latest
            if not exam_date:
                instances = get_inference_instances(request.patient_id, model_id=resolved_model_id)
                if instances:
                    exam_date = instances[0].get('exam_date')  # Get latest
                    logger.info(f"Using latest exam_date {exam_date} from instances")
        
        # Default to today's date if still no exam_date
        if not exam_date:
            exam_date = datetime.datetime.now().strftime("%Y%m%d")
            logger.warning(f"No exam_date found, using today: {exam_date}")

        disease_keys = [entry.get("key") for entry in getattr(patient, "diseases", []) if isinstance(entry, dict) and entry.get("key")]
        disease_key_set = set(disease_keys)
        alias_map = getattr(patient, "disease_alias_map", {}) or MODEL_CONFIGS.get(resolved_model_id, {}).get("disease_alias_map", {})
        
        # Store manual diagnosis data
        if request.manual_diagnosis or request.custom_diseases or request.diagnosis_notes or request.manual_descriptions:
            # Convert raw dict to ManualEyeDiagnosis objects
            processed_manual_diagnosis = {}
            if request.manual_diagnosis:
                lower_alias_map = {str(k).lower(): v for k, v in (alias_map or {}).items()}
                for eye_key, eye_data in request.manual_diagnosis.items():
                    if isinstance(eye_data, dict):
                        normalized_eye: Dict[str, bool] = {}
                        for raw_key, raw_val in eye_data.items():
                            canonical = (alias_map or {}).get(raw_key) or lower_alias_map.get(str(raw_key).lower(), raw_key)
                            if canonical and canonical not in disease_key_set:
                                disease_key_set.add(canonical)
                                disease_keys.append(canonical)
                            normalized_eye[canonical] = bool(raw_val)
                        for dk in disease_keys:
                            normalized_eye.setdefault(dk, False)
                        processed_manual_diagnosis[eye_key] = normalized_eye
                    else:
                        processed_manual_diagnosis[eye_key] = eye_data
            
            manual_data = ManualDiagnosisData(
                manual_diagnosis=processed_manual_diagnosis,
                custom_diseases=request.custom_diseases or CustomDiseases(),
                diagnosis_notes=request.diagnosis_notes or "",
                doctor_id=request.doctor_id,
                manual_descriptions=request.manual_descriptions or {
                    "left_eye": "",
                    "right_eye": ""
                }
            )
            
            # Store in memory with composite key
            storage_key = f"{request.patient_id}_{exam_date}"
            manual_diagnosis_storage[storage_key] = manual_data
            
            # Persist to disk with new structure
            with manual_diagnosis_file_lock:
                persisted = _load_manual_diagnosis_store()
                try:
                    payload = manual_data.dict()
                except Exception:
                    payload = {
                        "manual_diagnosis": manual_data.manual_diagnosis,
                        "custom_diseases": getattr(manual_data, "custom_diseases", {}),
                        "diagnosis_notes": getattr(manual_data, "diagnosis_notes", ""),
                        "doctor_id": getattr(manual_data, "doctor_id", None),
                        "manual_descriptions": getattr(manual_data, "manual_descriptions", {})
                    }
                
                # Ensure patient_id entry exists
                if request.patient_id not in persisted:
                    persisted[request.patient_id] = {}
                
                # Store under exam_date
                persisted[request.patient_id][exam_date] = payload
                _save_manual_diagnosis_store(persisted)
                
            print(f"Manual diagnosis data stored for patient: {request.patient_id}, exam_date: {exam_date}")
            print(f"Storage key: {storage_key}")
            print(f"Manual diagnosis: {manual_data.manual_diagnosis}")
            print(f"Custom diseases: {manual_data.custom_diseases}")
            print(f"Diagnosis notes: {manual_data.diagnosis_notes}")
        
        # Update image type/quality if provided
        if request.image_updates:
            for update_img in request.image_updates:
                for img in patient.eye_images:
                    if img.id == update_img["id"]:
                        img.type = update_img["type"]
                        img.quality = update_img["quality"]
            print(f"Image info updated for {request.patient_id}")
        
        return {"status": "Manual diagnosis and image info submitted successfully!", "exam_date": exam_date}


@app.post("/api/update_selection")
async def update_selection(request: UpdateSelectionRequest):
    """
    Recompute prediction_results and diagnosis_results based on the selected image IDs.
    
    NEW LOGIC: Images are selected by SLOT POSITION, not by their labeled type.
    Slot positions (0-indexed in the array):
    - Slot 0: 右眼CFP position
    - Slot 1: 左眼CFP position  
    - Slot 2: 右眼外眼照 position
    - Slot 3: 左眼外眼照 position
    
    The actual image type label may be incorrect, so we use the image's actual predictions
    based on slot position to determine which eye's results to update.
    """
    resolved_model_id = _resolve_model_id(request.model_id)
    cache_key = _make_patient_cache_key(resolved_model_id, request.patient_id, request.exam_date)
    
    with patient_data_lock:
        patient = patients_data_cache.get(cache_key)
        if not patient:
            # Fallback for older clients that might not send exam_date
            fallback_key = _make_patient_cache_key(resolved_model_id, request.patient_id, None)
            patient = patients_data_cache.get(fallback_key)
            if not patient:
                raise HTTPException(status_code=404, detail=f"Patient {cache_key} not found in cache. Load patient data first.")

        # We need access to original JSON probs to recompute.
        raw_cache_key = _make_raw_cache_key(resolved_model_id, request.patient_id, request.exam_date)
        if raw_cache_key not in raw_probs_cache:
            try:
                _warm_raw_cache_from_raw_json(resolved_model_id, request.patient_id, request.exam_date)
            except Exception:
                raise HTTPException(status_code=501, detail="Recompute not supported without source data")

        raw_by_id = raw_probs_cache[raw_cache_key]["by_id"]

        # Get selected images by slot position
        selected_ids = request.selected_image_ids or []
        
        # Map slot positions to expected types (for reference/fallback only)
        # Slot 0: 右眼CFP, Slot 1: 左眼CFP, Slot 2: 右眼外眼照, Slot 3: 左眼外眼照
        slot_config = [
            {"index": 0, "eye": "right", "type": "cfp", "label": "右眼CFP"},
            {"index": 1, "eye": "left", "type": "cfp", "label": "左眼CFP"},
            {"index": 2, "eye": "right", "type": "external", "label": "右眼外眼照"},
            {"index": 3, "eye": "left", "type": "external", "label": "左眼外眼照"}
        ]

        model_cfg = _get_model_config(resolved_model_id)
        alias_map = model_cfg.get("disease_alias_map", {})
        disease_keys = [entry["key"] for entry in model_cfg.get("diseases", [])]
        if not disease_keys:
            disease_keys = list((patient.prediction_results or {}).get("left_eye", {}).keys())

        # Initialize structures for each eye
        eye_data = {
            "left": {"cfp_probs": None, "ext_probs": None, "cfp_image_id": None, "ext_image_id": None},
            "right": {"cfp_probs": None, "ext_probs": None, "cfp_image_id": None, "ext_image_id": None}
        }

        # Process each slot position
        for slot in slot_config:
            if slot["index"] < len(selected_ids):
                image_id = selected_ids[slot["index"]]
                if image_id and image_id in raw_by_id:
                    raw_probs = raw_by_id[image_id]
                    mapped_probs: Dict[str, float] = {}
                    for k, v in (raw_probs or {}).items():
                        canonical = alias_map.get(k) or alias_map.get(str(k).lower()) or k
                        mapped_probs[canonical] = float(v)
                    
                    # Store probabilities based on slot position
                    eye = slot["eye"]
                    if slot["type"] == "cfp":
                        eye_data[eye]["cfp_probs"] = mapped_probs
                        eye_data[eye]["cfp_image_id"] = image_id
                    else:  # external
                        eye_data[eye]["ext_probs"] = mapped_probs
                        eye_data[eye]["ext_image_id"] = image_id

        # Build prediction results for each eye
        prediction_thresholds = dict(patient.prediction_thresholds or {})
        prediction_results: Dict[str, Dict[str, float]] = {}
        diagnosis_results = {}
        ext_cataract_used_overall = False
        debug_used = {"left_eye": {}, "right_eye": {}}

        for eye_name in ["left", "right"]:
            eye_key = f"{eye_name}_eye"
            data = eye_data[eye_name]
            
            # Start with zero probabilities
            probs = {k: 0.0 for k in disease_keys}
            
            # Use CFP probabilities if available (covers most diseases)
            if data["cfp_probs"]:
                probs.update(data["cfp_probs"])
                debug_used[eye_key]["cfp"] = data["cfp_image_id"]
            else:
                debug_used[eye_key]["cfp"] = "none"
            
            # Override cataract with external eye image if available
            ext_used = False
            if data["ext_probs"] and "白内障" in data["ext_probs"]:
                probs["白内障"] = data["ext_probs"]["白内障"]
                ext_used = True
                debug_used[eye_key]["ext"] = data["ext_image_id"]
            else:
                debug_used[eye_key]["ext"] = "none" if not data["ext_image_id"] else f"{data['ext_image_id']}_no_cataract"
            
            # Save predictions
            prediction_results[eye_key] = {k: float(probs.get(k, 0.0)) for k in disease_keys}
            
            # Thresholds and diagnoses
            eye_thresholds = dict(prediction_thresholds)
            if ext_used:
                eye_thresholds["白内障"] = CATARACT_EXTERNAL_THRESHOLD
                ext_cataract_used_overall = True
            
            # Build diagnosis using possibly adjusted threshold
            diag = {}
            for disease, threshold in eye_thresholds.items():
                prob_val = float(prediction_results[eye_key].get(disease, 0.0))
                diag[disease] = prob_val >= float(threshold)
            diagnosis_results[eye_key] = diag

        # If any eye used ext cataract, reflect that in thresholds object returned
        if ext_cataract_used_overall:
            prediction_thresholds["白内障"] = CATARACT_EXTERNAL_THRESHOLD

        # Update cache object
        patient.prediction_results = prediction_results
        patient.diagnosis_results = diagnosis_results
        patient.prediction_thresholds = prediction_thresholds

        # Convert Pydantic models to dictionaries for JSON response
        return {
            "status": "Selection updated",
            "prediction_results": prediction_results,
            "diagnosis_results": diagnosis_results,
            "prediction_thresholds": prediction_thresholds,
            "debug_used_images": debug_used,
        }

@app.get("/api/manual_diagnosis/{ris_exam_id}")
async def get_manual_diagnosis(ris_exam_id: str, exam_date: Optional[str] = None):
    """
    Returns the manual diagnosis data for a specific patient and exam.
    If exam_date is not provided, tries to find the latest exam.
    """
    with patient_data_lock:
        # Try with exam_date if provided
        if exam_date:
            cache_key = f"{ris_exam_id}_{exam_date}"
            manual_data = manual_diagnosis_storage.get(cache_key)
            if manual_data:
                if getattr(manual_data, "manual_descriptions", None) is None:
                    manual_data.manual_descriptions = {"left_eye": "", "right_eye": ""}
                return manual_data
        
        # Try without exam_date (backward compatibility)
        manual_data = manual_diagnosis_storage.get(ris_exam_id)
        if manual_data:
            if getattr(manual_data, "manual_descriptions", None) is None:
                manual_data.manual_descriptions = {"left_eye": "", "right_eye": ""}
            return manual_data
        
        # If no exam_date provided, try to find any exam for this patient
        if not exam_date:
            # Look for any key starting with this patient_id
            for key, data in manual_diagnosis_storage.items():
                if key.startswith(f"{ris_exam_id}_"):
                    logger.info(f"Found manual diagnosis for {key}")
                    if getattr(data, "manual_descriptions", None) is None:
                        data.manual_descriptions = {"left_eye": "", "right_eye": ""}
                    return data
        
        # Return empty structure if no manual diagnosis exists yet
        from datatype import ManualDiagnosisData, CustomDiseases
        return ManualDiagnosisData(
            manual_diagnosis={},
            custom_diseases=CustomDiseases(),
            diagnosis_notes="",
            doctor_id=None,
            manual_descriptions={"left_eye": "", "right_eye": ""}
        )

@app.get("/api/manual_diagnoses")
async def get_all_manual_diagnoses():
    """
    Returns all stored manual diagnosis data for debugging purposes.
    """
    with patient_data_lock:
        return {
            "stored_diagnoses": list(manual_diagnosis_storage.keys()),
            "data": manual_diagnosis_storage
        }

@app.post("/api/add_new_patient")
async def add_new_patient_data(patient_data: PatientData):
    """
    Adds a new patient's data received from an external source (e.g., trigger script)
    and caches it for future access.
    """
    with patient_data_lock:
        resolved_model_id = _resolve_model_id(patient_data.model_id)
        cache_key = _make_patient_cache_key(resolved_model_id, patient_data.patient_id, None)
        if cache_key in patients_data_cache:
            print(f"Patient {patient_data.patient_id} (model {resolved_model_id}) already exists in cache. Overwriting.")
        
        patients_data_cache[cache_key] = patient_data
        print(f"New patient {patient_data.patient_id} cached for model {resolved_model_id}.")
        return {"status": f"Patient {patient_data.patient_id} cached successfully for model {resolved_model_id}."}


@app.post("/api/alter_threshold")
async def alter_threshold(request: AlterThresholdRequest):
    """
    Switch between threshold sets defined by the active model and recompute diagnoses.
    """
    resolved_model_id = _resolve_model_id(request.model_id)
    model_cfg = _get_model_config(resolved_model_id)
    cache_key = _make_patient_cache_key(resolved_model_id, request.patient_id, request.exam_date)
    threshold_sets = model_cfg["threshold_sets"]
    threshold_indices = model_cfg["threshold_set_indices"]

    with patient_data_lock:
        patient = patients_data_cache.get(cache_key)
        if not patient:
            # Fallback to latest if specific exam not cached
            fallback_key = _make_patient_cache_key(resolved_model_id, request.patient_id, None)
            patient = patients_data_cache.get(fallback_key)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not loaded for model {resolved_model_id}.")

        current_set_id = getattr(patient, "active_threshold_set_id", None)
        current_idx = threshold_indices.get(current_set_id, getattr(patient, "active_threshold_set_index", 0))

        if request.threshold_set_id and request.threshold_set_id in threshold_indices:
            target_set_id = request.threshold_set_id
            target_idx = threshold_indices[target_set_id]
        else:
            target_idx = (current_idx + 1) % len(threshold_sets)
            target_set_id = threshold_sets[target_idx]["id"]

        new_thresholds = threshold_sets[target_idx]["values"]

        patient.prediction_thresholds = dict(new_thresholds)
        patient.active_threshold_set_id = target_set_id
        patient.active_threshold_set_index = target_idx
        patient.active_threshold_set = target_idx  # backwards compatibility
        patient.threshold_sets = threshold_sets
        patient.model_id = resolved_model_id
        patient.model_name = model_cfg["name"]
        patient.diseases = model_cfg["diseases"]

        updated_diagnosis_results: Dict[str, Dict[str, bool]] = {}
        for eye_key, probs in (patient.prediction_results or {}).items():
            diag_flags: Dict[str, bool] = {}
            for disease_key, threshold_value in new_thresholds.items():
                prob_val = float((probs or {}).get(disease_key, 0.0))
                diag_flags[disease_key] = prob_val >= float(threshold_value)
            updated_diagnosis_results[eye_key] = diag_flags

        patient.diagnosis_results = updated_diagnosis_results

        return {
            "status": "Threshold altered successfully",
            "active_threshold_set": target_idx,
            "active_threshold_set_id": target_set_id,
            "threshold_sets": threshold_sets,
            "new_thresholds": new_thresholds,
            "updated_diagnosis_results": updated_diagnosis_results,
        }


# --- Consultation Data Models (accept free text and arrays) ---
class EyeSymptomData(BaseModel):
    mainSymptom: Optional[str] = None
    onsetMethod: Optional[str] = None
    onsetTime: Optional[str] = None
    accompanyingSymptoms: Optional[Union[List[str], str]] = None  # allow string or list
    medicalHistory: Optional[str] = None
    mainSymptomOther: Optional[str] = None

    class Config:
        extra = 'allow'  # tolerate extra keys from UI

class ConsultationData(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    affectedArea: Optional[List[str]] = None
    leftEye: Optional[EyeSymptomData] = None
    rightEye: Optional[EyeSymptomData] = None
    bothEyes: Optional[EyeSymptomData] = None
    submissionTime: Optional[str] = None

    class Config:
        extra = 'allow'  # tolerate extra keys

class SaveConsultationRequest(BaseModel):
    ris_exam_id: str  # 修改为ris_exam_id
    consultation_data: ConsultationData



# --- Helpers for consultation file I/O ---
def load_consultation_data() -> List[Dict[str, Any]]:
    if not os.path.exists(CONSULTATION_DATA_PATH):
        return []
    with open(CONSULTATION_DATA_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # If file accidentally not a list, coerce to list
            return [data]
        except json.JSONDecodeError:
            return []

def save_consultation_data(data: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(CONSULTATION_DATA_PATH), exist_ok=True)
    with open(CONSULTATION_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def _to_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        return "、".join([str(x) for x in v])
    return str(v)

def normalize_consultation_text(entry: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure eye sections store free text where needed
    for eye_field in ("leftEye", "rightEye", "bothEyes"):
        eye = entry.get(eye_field)
        if isinstance(eye, dict):
            if "accompanyingSymptoms" in eye:
                eye["accompanyingSymptoms"] = _to_text(eye.get("accompanyingSymptoms"))
    return entry


def _find_consultation_by_exam_id(all_data: List[Dict[str, Any]], ris_exam_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Locate a consultation entry (refined preferred) linked to a specific ris_exam_id.

    Returns (matched_consultation, base_entry) where base_entry is the original list item
    containing the consultation (useful for fallback metadata such as submissionTime).
    """
    if not ris_exam_id:
        return None, None

    for entry in all_data or []:
        if not isinstance(entry, dict):
            continue

        refined = entry.get("refined")
        if isinstance(refined, dict) and refined.get("_exam_ris_exam_id") == ris_exam_id:
            return refined, entry

        if entry.get("_exam_ris_exam_id") == ris_exam_id:
            return entry, entry

    return None, None

def _parse_iso_to_aware_dt(s: Optional[str]) -> Optional[datetime.datetime]:
    """
    Parse ISO string into a timezone-aware datetime.
    - Accepts 'Z' suffix or explicit offsets.
    - If naive (no tz), assume local timezone.
    """
    if not s:
        return None
    try:
        dt = datetime.datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    except Exception:
        return None
    if dt.tzinfo is None:
        try:
            local_tz = datetime.datetime.now().astimezone().tzinfo
        except Exception:
            local_tz = datetime.timezone.utc
        dt = dt.replace(tzinfo=local_tz)
    return dt

# --- Correct matching: submissionTime should be BEFORE or equal to examineTime, choose closest ---
def find_best_matching_consultation(all_data: List[Dict[str, Any]], patient_name: Optional[str], exam_time: Optional[str]):
    if not patient_name:
        return None

    # Parse exam_time -> aware (UTC for comparison)
    exam_dt = _parse_iso_to_aware_dt(exam_time)
    exam_dt_utc = exam_dt.astimezone(datetime.timezone.utc) if exam_dt else None

    # Filter by name
    same_name = [x for x in all_data if x.get("name") == patient_name]
    if not same_name:
        return None

    if not exam_dt_utc:
        # If no exam time, return the last record by submissionTime if present
        def time_key(x):
            st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
            return (st_dt.astimezone(datetime.timezone.utc) if st_dt else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))
        return sorted(same_name, key=time_key)[-1]

    # Choose the entry with submissionTime <= exam_time and closest to exam_time
    candidates = []
    for x in same_name:
        st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st_dt:
            continue
        st_dt_utc = st_dt.astimezone(datetime.timezone.utc)
        if st_dt_utc <= exam_dt_utc:
            candidates.append((x, st_dt_utc))
    if candidates:
        candidates.sort(key=lambda t: (exam_dt_utc - t[1]))
        return candidates[0][0]

    # Fallback: if none before exam, return the earliest after exam (closest)
    after = []
    for x in same_name:
        st_dt = _parse_iso_to_aware_dt(x.get("submissionTime"))
        if not st_dt:
            continue
        st_dt_utc = st_dt.astimezone(datetime.timezone.utc)
        if st_dt_utc > exam_dt_utc:
            after.append((x, st_dt_utc))
    if after:
        after.sort(key=lambda t: (t[1] - exam_dt_utc))
        return after[0][0]

    return None

# --- Helpers for patient context ---
def _get_patient_context(ris_exam_id: str, model_id: Optional[str] = None, exam_date: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Returns {'name': str|None, 'examineTime': str|None} from cache, else from aggregated inference data.
    """
    ctx = {"name": None, "examineTime": None}
    resolved_model_id = _resolve_model_id(model_id)
    try:
        cache_key = _make_patient_cache_key(resolved_model_id, ris_exam_id, exam_date)
        patient = patients_data_cache.get(cache_key)
        if not patient and exam_date:
            fallback_key = _make_patient_cache_key(resolved_model_id, ris_exam_id, None)
            patient = patients_data_cache.get(fallback_key)
        if patient:
            ctx["name"] = getattr(patient, "name", None)
            ctx["examineTime"] = getattr(patient, "examine_time", None)
            if ctx["name"] or ctx["examineTime"]:
                return ctx
    except Exception:
        pass

    # Fallback to aggregated inference cache
    try:
        record = get_inference_record(ris_exam_id, exam_date, model_id=resolved_model_id)
        if isinstance(record, dict):
            ctx["name"] = record.get("name")
            ctx["examineTime"] = record.get("examineTime")
    except Exception:
        pass

    if not ctx["name"] or not ctx["examineTime"]:
        try:
            with consultation_data_lock:
                all_cons = load_consultation_data()
                matched, base_entry = _find_consultation_by_exam_id(all_cons, ris_exam_id)
            source = matched or base_entry
            if isinstance(source, dict):
                if not ctx["name"] and source.get("name"):
                    ctx["name"] = source.get("name")
                if not ctx["examineTime"]:
                    ctx["examineTime"] = source.get("examineTime") or source.get("submissionTime")
        except Exception:
            pass
    return ctx

def _resolve_consultation_time(entry: Dict[str, Any]) -> Optional[datetime.datetime]:
    """Prefer refined timestamps, fallback to base submissionTime."""
    if not isinstance(entry, dict):
        return None

    refined = entry.get("refined")
    if isinstance(refined, dict):
        for key in ("refinedTime", "submissionTime"):
            dt = _parse_iso_to_aware_dt(refined.get(key))
            if dt:
                return dt.astimezone(datetime.timezone.utc)

    dt = _parse_iso_to_aware_dt(entry.get("submissionTime"))
    if dt:
        return dt.astimezone(datetime.timezone.utc)
    return None


def _find_best_matching_index(all_data: List[Dict[str, Any]], patient_name: Optional[str], exam_time: Optional[str]) -> Optional[int]:
    """
    Same selection rule as find_best_matching_consultation, but returns index in list.
    Prefers submission/refined time <= examTime (closest), else earliest after.
    Matches strictly by the provided patient_name (including refined name field).
    """
    normalized_name = (patient_name or "").strip()
    if not normalized_name or not isinstance(all_data, list) or len(all_data) == 0:
        return None

    exam_dt = _parse_iso_to_aware_dt(exam_time)
    exam_utc = exam_dt.astimezone(datetime.timezone.utc) if exam_dt else None

    name_idxs: List[Tuple[int, Dict[str, Any]]] = []
    for i, entry in enumerate(all_data):
        if not isinstance(entry, dict):
            continue
        entry_names: List[str] = []
        base_name = entry.get("name")
        if base_name:
            entry_names.append(str(base_name).strip())
        refined = entry.get("refined")
        if isinstance(refined, dict):
            refined_name = refined.get("name")
            if refined_name:
                entry_names.append(str(refined_name).strip())

        if normalized_name and normalized_name in entry_names:
            name_idxs.append((i, entry))

    if not name_idxs:
        return None

    if not exam_utc:
        # no exam time -> pick latest by available timestamps
        def time_key(entry: Dict[str, Any]):
            dt = _resolve_consultation_time(entry)
            return dt if dt else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

        name_idxs.sort(key=lambda t: time_key(t[1]))
        return name_idxs[-1][0]

    # before or equal to exam (closest)
    candidates: List[Tuple[int, datetime.datetime]] = []
    for idx, entry in name_idxs:
        st_utc = _resolve_consultation_time(entry)
        if not st_utc:
            continue
        if st_utc <= exam_utc:
            candidates.append((idx, st_utc))
    if candidates:
        candidates.sort(key=lambda t: (exam_utc - t[1]))
        return candidates[0][0]

    # earliest after exam
    after: List[Tuple[int, datetime.datetime]] = []
    for idx, entry in name_idxs:
        st_utc = _resolve_consultation_time(entry)
        if not st_utc:
            continue
        if st_utc > exam_utc:
            after.append((idx, st_utc))
    if after:
        after.sort(key=lambda t: (t[1] - exam_utc))
        return after[0][0]
    return None

# --- API: get/save consultation ---
@app.get("/api/consultation/{ris_exam_id}")
async def get_consultation_info(ris_exam_id: str, patient_name: Optional[str] = None):
    """
    获取问诊信息
    - 如果提供patient_name参数，优先使用该名称搜索问诊信息
    - 否则使用ris_exam_id对应的患者名称搜索
    """
    # 获取患者上下文
    ctx = _get_patient_context(ris_exam_id)
    
    # 决定使用哪个患者姓名进行搜索
    search_name = patient_name if patient_name else ctx.get("name")
    examine_time = ctx.get("examineTime")

    with consultation_data_lock:
        all_consultations = load_consultation_data()
        
        # 如果提供了patient_name，返回所有同名的问诊记录供选择
        if patient_name:
            same_name_consultations = []
            for i, consultation in enumerate(all_consultations):
                if consultation.get("name") == patient_name:
                    # 构建显示信息
                    base_info = {
                        "index": i,
                        "name": consultation.get("name"),
                        "age": consultation.get("age"),
                        "gender": consultation.get("gender"),
                        "phone": consultation.get("phone"),
                        "submissionTime": consultation.get("submissionTime"),
                        "hasRefined": "refined" in consultation and isinstance(consultation["refined"], dict)
                    }
                    
                    # 如果有refined数据，也添加refined的信息
                    if base_info["hasRefined"]:
                        refined = consultation["refined"]
                        refined_info = base_info.copy()
                        refined_info.update({
                            "age": refined.get("age", base_info["age"]),
                            "gender": refined.get("gender", base_info["gender"]),
                            "phone": refined.get("phone", base_info["phone"]),
                            "submissionTime": refined.get("refinedTime", refined.get("submissionTime", base_info["submissionTime"])),
                            "isRefined": True
                        })
                        same_name_consultations.append(refined_info)
                    else:
                        base_info["isRefined"] = False
                        same_name_consultations.append(base_info)
            
            # 按提交时间排序
            same_name_consultations.sort(
                key=lambda x: _parse_iso_to_aware_dt(x.get("submissionTime")) or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc),
                reverse=True
            )
            
            return {
                "consultation_data": None,
                "status": "multiple_matches",
                "same_name_consultations": same_name_consultations,
                "search_name": patient_name,
                "source": "questionnaire_data.json",
                "path": CONSULTATION_DATA_PATH,
            }
        
        # 原有逻辑：根据名称和时间匹配最佳问诊记录
        idx = _find_best_matching_index(all_consultations, search_name, examine_time)
        if idx is None:
            return {
                "consultation_data": None,
                "status": "no_match",
                "source": "questionnaire_data.json",
                "path": CONSULTATION_DATA_PATH,
            }

        base = all_consultations[idx]
        refined = base.get("refined")
        # 优先使用refined数据
        if isinstance(refined, dict):
            result = dict(refined)
            result.setdefault("name", base.get("name"))
            # 显示refinedTime作为submissionTime
            if "refinedTime" in result:
                result.setdefault("submissionTime", result.get("refinedTime"))
            else:
                result.setdefault("submissionTime", base.get("submissionTime"))
            return {
                "consultation_data": result,
                "status": "success_refined",
                "source": f"questionnaire_data.json[{idx}].refined",
                "path": CONSULTATION_DATA_PATH
            }
        else:
            # 返回原始数据
            minimal = {
                "name": base.get("name"),
                "age": base.get("age"),
                "gender": base.get("gender"),
                "phone": base.get("phone"),
                "affectedArea": base.get("affectedArea"),
                "leftEye": base.get("leftEye"),
                "rightEye": base.get("rightEye"),
                "bothEyes": base.get("bothEyes"),
                "submissionTime": base.get("submissionTime"),
            }
            return {
                "consultation_data": minimal,
                "status": "success_original",
                "source": f"questionnaire_data.json[{idx}]",
                "path": CONSULTATION_DATA_PATH
            }

### 2. 修改根据索引获取特定问诊记录的API

@app.get("/api/consultation/{ris_exam_id}/by_index/{consultation_index}")
async def get_consultation_by_index(ris_exam_id: str, consultation_index: int, use_refined: bool = True):
    """
    根据索引获取特定的问诊记录
    """
    with consultation_data_lock:
        all_consultations = load_consultation_data()
        
        if consultation_index < 0 or consultation_index >= len(all_consultations):
            raise HTTPException(status_code=404, detail="Consultation index out of range")
        
        base = all_consultations[consultation_index]
        
        if use_refined and "refined" in base and isinstance(base["refined"], dict):
            # 使用refined数据
            result = dict(base["refined"])
            result.setdefault("name", base.get("name"))
            if "refinedTime" in result:
                result.setdefault("submissionTime", result.get("refinedTime"))
            else:
                result.setdefault("submissionTime", base.get("submissionTime"))
            return {
                "consultation_data": result,
                "status": "success_refined",
                "source": f"questionnaire_data.json[{consultation_index}].refined",
                "path": CONSULTATION_DATA_PATH
            }
        else:
            # 使用原始数据
            minimal = {
                "name": base.get("name"),
                "age": base.get("age"),
                "gender": base.get("gender"),
                "phone": base.get("phone"),
                "affectedArea": base.get("affectedArea"),
                "leftEye": base.get("leftEye"),
                "rightEye": base.get("rightEye"),
                "bothEyes": base.get("bothEyes"),
                "submissionTime": base.get("submissionTime"),
            }
            return {
                "consultation_data": minimal,
                "status": "success_original",
                "source": f"questionnaire_data.json[{consultation_index}]",
                "path": CONSULTATION_DATA_PATH
            }

# 保存问诊信息的API也需要修改参数名
@app.post("/api/consultation")
async def save_consultation_info(request: SaveConsultationRequest):
    # 需要修改SaveConsultationRequest中的字段名
    with consultation_data_lock:
        all_consultations = load_consultation_data()

        # Build normalized refined payload
        refined_payload = request.consultation_data.dict(exclude_none=True)
        refined_payload = normalize_consultation_text(refined_payload)

        # Ensure name if missing
        if not refined_payload.get("name"):
            ctx = _get_patient_context(request.ris_exam_id)  # 修改为ris_exam_id
            if ctx.get("name"):
                refined_payload["name"] = ctx["name"]

        # Timestamp for refined save (tz-aware)
        refined_time = datetime.datetime.now().astimezone().isoformat()
        refined_payload["refinedTime"] = refined_time
        refined_payload["_exam_ris_exam_id"] = request.ris_exam_id  # 修改字段名

        # Locate original questionnaire item by name + examineTime alignment
        ctx = _get_patient_context(request.ris_exam_id)  # 修改为ris_exam_id
        idx = _find_best_matching_index(all_consultations, ctx.get("name"), ctx.get("examineTime"))

        if idx is not None:
            # Overwrite previous refined content
            all_consultations[idx]["refined"] = refined_payload
        else:
            # Fallback: create a minimal container with refined inside
            container = {
                "name": refined_payload.get("name"),
                "submissionTime": refined_time,
                "refined": refined_payload,
            }
            all_consultations.append(container)

        save_consultation_data(all_consultations)

    return {
        "status": "Consultation data refined and saved successfully",
        "refined": refined_payload,
        "path": CONSULTATION_DATA_PATH
    }


# --- LLM prompts config (provider/base/model stay in .env) ---
LLM_PROMPTS_PATH_ENV = os.getenv("LLM_PROMPTS_PATH", "config/llm_prompts.json")
if not os.path.isabs(LLM_PROMPTS_PATH_ENV):
    LLM_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), LLM_PROMPTS_PATH_ENV)
else:
    LLM_PROMPTS_PATH = LLM_PROMPTS_PATH_ENV

def load_llm_prompts() -> Dict[str, Any]:
    defaults = {
        "system_prompt": "",
        "update_prompt": "请基于最新问诊信息、AI预测与人工复检结果，生成简要且可操作的临床意见摘要。",
        "include_patient_context": True,
        "context_template": "患者：{patient_name}（{age}岁，{gender_zh}），检查时间：{examine_time}\n问诊要点：{consultation_summary}\nAI预测（左眼）：{ai_left_summary}\nAI预测（右眼）：{ai_right_summary}\n人工复核：{manual_summary}\n阈值：{threshold_summary}"
    }
    try:
        p = Path(LLM_PROMPTS_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {**defaults, **data}
    except Exception:
        pass
    return defaults

@app.get("/api/llm_config")
async def get_llm_config():
    cfg = load_llm_prompts()
    # Only prompts/flags; provider/base/model remain in .env
    return {
        "system_prompt": cfg.get("system_prompt", ""),
        "update_prompt": cfg.get("update_prompt", ""),
        "include_patient_context": cfg.get("include_patient_context", True),
        "has_context_template": bool(cfg.get("context_template"))
    }

def _zh_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    s = str(g).strip().lower()
    if s == "male": return "男"
    if s == "female": return "女"
    if s == "other": return "其他"
    # 处理中文输入
    if s in ["男", "女性", "女", "其他"]:
        return s if s in ["男", "女", "其他"] else "女"
    return g

def _format_prob(p: Optional[float]) -> str:
    try:
        return f"{float(p):.2f}"
    except Exception:
        return "-"

def _summarize_consultation(cons: Optional[Dict[str, Any]]) -> str:
    if not cons or not isinstance(cons, dict):
        return "无"
    
    map_area = lambda a: "左眼" if a=="left" else ("右眼" if a=="right" else "双眼")
    areas = [map_area(a) for a in (cons.get("affectedArea") or [])]
    parts = []
    if areas: parts.append(f"受累部位：{'、'.join(areas)}")
    def eye_part(label, obj):
        if not obj: return None
        ms = obj.get("mainSymptom")
        om = obj.get("onsetMethod")
        ot = obj.get("onsetTime")
        ac = obj.get("accompanyingSymptoms")
        mh = obj.get("medicalHistory")
        segs = []
        if ms: segs.append(f"主要：{ms}")
        if om or ot:
            # Use onset method text directly (no mapping needed)
            onset_desc = om if om else ""
            time_desc = ot if ot else ""
            combined = f"{onset_desc} {time_desc}".strip()
            if combined:
                segs.append(f"起病：{combined}")
        if ac: segs.append(f"伴随：{ac}")
        if mh: segs.append(f"病史：{mh}")
        return f"{label}（" + "；".join(segs) + "）" if segs else None
    for label, key in (("左眼","leftEye"),("右眼","rightEye"),("双眼","bothEyes")):
        x = eye_part(label, cons.get(key))
        if x: parts.append(x)
    return "；".join(parts) if parts else "无"

def _summarize_thresholds(patient: Optional[PatientData]) -> str:
    """总结阈值设置"""
    if not patient:
        return "使用默认阈值"
    
    th = getattr(patient, "prediction_thresholds", None)
    if not th:
        return "AMD:0.30；青光眼:0.60；糖网:0.50；白内障:0.60"
    
    thd = th.dict() if hasattr(th, "dict") else th
    if not isinstance(thd, dict):
        return "AMD:0.30；青光眼:0.60；糖网:0.50；白内障:0.60"
    
    items = []
    disease_mapping = {
        "年龄相关性黄斑变性": "AMD",
        "青光眼": "青光眼", 
        "糖尿病性视网膜病变": "糖网",
        "白内障": "白内障"
    }
    
    for disease, short_name in disease_mapping.items():
        threshold = thd.get(disease, 0.5)
        items.append(f"{short_name}:{threshold:.2f}")
    
    return "；".join(items)

def _summarize_manual(patient_id: str, exam_date: Optional[str] = None) -> str:
    """
    Summarize manual diagnosis for a patient.
    Tries composite key first, then falls back to simple patient_id for backward compatibility.
    """
    md = None
    
    # Try with exam_date if provided
    if exam_date:
        cache_key = f"{patient_id}_{exam_date}"
        md = manual_diagnosis_storage.get(cache_key)
    
    # Fall back to simple patient_id (backward compatibility)
    if not md:
        md = manual_diagnosis_storage.get(patient_id)
    
    # If still no data, try to find any exam for this patient
    if not md and not exam_date:
        for key, data in manual_diagnosis_storage.items():
            if key.startswith(f"{patient_id}_"):
                md = data
                break
    
    if not md:
        return "无"
    
    try:
        md_dict = md.dict()
    except Exception:
        md_dict = {
            "manual_diagnosis": getattr(md, "manual_diagnosis", {}),
            "custom_diseases": getattr(md, "custom_diseases", {}),
            "diagnosis_notes": getattr(md, "diagnosis_notes", "")
        }
    pos = [k for k, v in (md_dict.get("manual_diagnosis") or {}).items() if v]
    extra = md_dict.get("custom_diseases") or {}
    notes = (md_dict.get("diagnosis_notes") or "").strip()
    segs = []
    if pos: segs.append("人工判断：" + "、".join(pos))
    if getattr(extra, "left_eye", None) or getattr(extra, "right_eye", None):
        # pydantic object support
        l = getattr(extra, "left_eye", "")
        r = getattr(extra, "right_eye", "")
        if str(l).strip(): segs.append(f"左眼附加：{l}")
        if str(r).strip(): segs.append(f"右眼附加：{r}")
    elif isinstance(extra, dict):
        if extra.get("left_eye"): segs.append(f"左眼附加：{extra.get('left_eye')}")
        if extra.get("right_eye"): segs.append(f"右眼附加：{extra.get('right_eye')}")
    if notes: segs.append(f"备注：{notes}")
    return "；".join(segs) if segs else "无"

def _summarize_ai_detailed(patient: Optional[PatientData], eye_key: str) -> str:
    """改进的AI预测总结，直接显示疾病可能性判断结果"""
    if not patient:
        return "无预测数据"
    
    # 获取预测结果和阈值
    prediction_results = getattr(patient, "prediction_results", None)
    prediction_thresholds = getattr(patient, "prediction_thresholds", None)
    diagnosis_results = getattr(patient, "diagnosis_results", None)
    
    if not prediction_results:
        # 如果没有prediction_results，尝试从原始数据重新计算
        logger.warning(f"No prediction_results for patient {patient.patient_id}, trying to use raw data")
        return _fallback_ai_summary(patient, eye_key)
    
    # 获取该眼的预测概率
    eye_probs = None
    if hasattr(prediction_results, eye_key):
        eye_probs = getattr(prediction_results, eye_key)
    elif isinstance(prediction_results, dict):
        eye_probs = prediction_results.get(eye_key)
    
    if hasattr(eye_probs, "dict"):
        eye_probs = eye_probs.dict()
    elif hasattr(eye_probs, "__dict__"):
        eye_probs = eye_probs.__dict__
    
    if not isinstance(eye_probs, dict) or not eye_probs:
        logger.warning(f"No eye_probs for {eye_key}, trying fallback")
        return _fallback_ai_summary(patient, eye_key)
    
    # 获取阈值
    thresholds = {}
    if prediction_thresholds:
        if hasattr(prediction_thresholds, "dict"):
            thresholds = prediction_thresholds.dict()
        elif hasattr(prediction_thresholds, "__dict__"):
            thresholds = prediction_thresholds.__dict__
        elif isinstance(prediction_thresholds, dict):
            thresholds = prediction_thresholds
    
    # 默认阈值
    default_thresholds = {
        "年龄相关性黄斑变性": 0.30,
        "青光眼": 0.60,
        "糖尿病性视网膜病变": 0.50,
        "白内障": 0.60,
        "其他黄斑病变": 0.50,
        "其他视网膜病": 0.50,
        "视网膜静脉阻塞（RVO）": 0.50,
        "视网膜动脉阻塞（RAO）": 0.50,
        "病理性近视": 0.50,
        "视网膜脱离（RD）": 0.50,
        "正常": 0.50
    }
    
    # 合并阈值，优先使用设定的阈值
    final_thresholds = {**default_thresholds, **thresholds}
    
    # 获取诊断结果
    eye_diagnosis = {}
    if diagnosis_results:
        if hasattr(diagnosis_results, eye_key):
            diag_data = getattr(diagnosis_results, eye_key)
            if hasattr(diag_data, "dict"):
                eye_diagnosis = diag_data.dict()
            elif hasattr(diag_data, "__dict__"):
                eye_diagnosis = diag_data.__dict__
            elif isinstance(diag_data, dict):
                eye_diagnosis = diag_data
        elif isinstance(diagnosis_results, dict):
            diag_data = diagnosis_results.get(eye_key, {})
            if hasattr(diag_data, "dict"):
                eye_diagnosis = diag_data.dict()
            elif isinstance(diag_data, dict):
                eye_diagnosis = diag_data
    
    # 按预测值降序排序，只显示前几个主要疾病
    try:
        sorted_diseases = sorted(eye_probs.items(), key=lambda x: float(x[1]), reverse=True)
    except Exception:
        logger.error(f"Error sorting diseases for {eye_key}: {eye_probs}")
        return "预测数据格式错误"
    
    items = []
    for disease, prob in sorted_diseases[:5]:  # 只显示前5个疾病
        try:
            prob_val = float(prob)
            threshold = float(final_thresholds.get(disease, 0.5))
            is_positive = eye_diagnosis.get(disease, prob_val > threshold)
            
            # 根据是否超过阈值显示不同格式
            if is_positive:
                items.append(f"{disease}: **阳性** (预测值{prob_val:.3f} > 阈值{threshold:.3f})")
            else:
                items.append(f"{disease}: 阴性 (预测值{prob_val:.3f} ≤ 阈值{threshold:.3f})")
        except Exception as e:
            logger.warning(f"Error processing disease {disease}: {e}")
            continue
    
    return "；".join(items) if items else "无有效预测"

def _fallback_ai_summary(patient: Optional[PatientData], eye_key: str) -> str:
    """当无法从patient对象获取预测结果时的fallback方法"""
    if not patient:
        return "无预测数据"
    
    patient_id = getattr(patient, "patient_id", None)
    if not patient_id:
        return "无患者ID"
    
    # 尝试从原始缓存数据中获取
    try:
        raw_cache = raw_probs_cache.get(patient_id)
        if not raw_cache:
            return "无原始预测缓存"
        
        # 根据eye_key确定查找的图像类型
        cfp_type = "左眼CFP" if eye_key == "left_eye" else "右眼CFP"
        
        # 获取该类型的最新图像数据
        raw_by_type = raw_cache.get("by_type", {})
        images_of_type = raw_by_type.get(cfp_type, [])
        
        if not images_of_type:
            return f"无{cfp_type}数据"
        
        # 使用最新的图像预测结果
        latest_image = images_of_type[0]  # 已经按时间排序
        probs = latest_image.get("probs", {})
        
        if not probs:
            return "无概率数据"
        
        # 疾病名称映射
        diagnosis_mapping = {
            "青光眼": "青光眼",
            "糖尿病性视网膜病变": "糖网",
            "年龄相关性黄斑变性": "AMD", 
            "病理性近视": "病理性近视",
            "视网膜静脉阻塞（RVO）": "RVO",
            "视网膜动脉阻塞（RAO）": "RAO",
            "视网膜脱离（RD）": "视网膜脱离",
            "其他视网膜病": "其它视网膜病",
            "其他黄斑病变": "其它黄斑病变",
            "白内障": "白内障",
            "正常": "正常"
        }
        
        # 转换疾病名称并排序
        mapped_probs = {}
        for orig_name, prob in probs.items():
            mapped_name = diagnosis_mapping.get(orig_name, orig_name)
            mapped_probs[mapped_name] = float(prob)
        
        sorted_diseases = sorted(mapped_probs.items(), key=lambda x: x[1], reverse=True)
        
        # 使用默认阈值
        default_thresholds = {
            "AMD": 0.30,
            "青光眼": 0.60,
            "糖网": 0.50,
            "白内障": 0.60,
        }
        
        items = []
        for disease, prob in sorted_diseases[:5]:
            threshold = default_thresholds.get(disease, 0.5)
            is_positive = prob > threshold
            
            if is_positive:
                items.append(f"{disease}: **阳性** (预测值{prob:.3f} > 阈值{threshold:.3f})")
            else:
                items.append(f"{disease}: 阴性 (预测值{prob:.3f} ≤ 阈值{threshold:.3f})")
        
        return "；".join(items) if items else "无有效预测"
        
    except Exception as e:
        logger.error(f"Fallback AI summary failed: {e}")
        return f"获取预测数据失败: {str(e)}"

def _build_context_placeholders(
    ris_exam_id: Optional[str],
    patient_name: Optional[str] = None,
    exam_date: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build context placeholders for LLM prompts.
    
    Args:
        ris_exam_id: Patient exam ID
        patient_name: Optional patient name to prioritize for consultation matching (from URL param)
        exam_date: Optional exam date to retrieve specific manual diagnosis
    """
    resolved_model_id, p = (None, None)
    if ris_exam_id:
        resolved_model_id, p = _get_cached_patient(ris_exam_id, model_id=model_id, exam_date=exam_date)
    
    # 从患者数据中获取基本信息
    name = getattr(p, "name", None) if p else None
    age = getattr(p, "age", None) if p and hasattr(p, "age") else None
    gender = getattr(p, "gender", None) if p and hasattr(p, "gender") else None
    examine_time = getattr(p, "examine_time", None) if p else None
    
    lookup_model = _resolve_model_id(model_id) if model_id else (resolved_model_id or DEFAULT_MODEL_ID)

    # 如果患者数据中没有年龄和性别，尝试从推理结果缓存中获取
    if not age or not gender or not name or not examine_time:
        try:
            record = get_inference_record(ris_exam_id or "", exam_date, model_id=lookup_model)
            if isinstance(record, dict):
                if not age and record.get("age") is not None:
                    age = record.get("age")
                if not gender and record.get("gender") is not None:
                    gender = record.get("gender")
                if not name and record.get("name"):
                    name = record.get("name")
                if not examine_time and record.get("examineTime"):
                    examine_time = record.get("examineTime")
        except Exception as e:
            logger.warning(f"Failed to load additional patient info: {str(e)}")

    ctx = _get_patient_context(ris_exam_id or "", model_id=lookup_model, exam_date=exam_date)

    # 从问诊数据中获取年龄和性别信息（优先级更高，因为这是用户填写的最新信息）
    cons: Optional[Dict[str, Any]] = None
    base_entry: Optional[Dict[str, Any]] = None
    
    # PRIORITY 1: Use patient_name from URL parameter if provided
    # PRIORITY 2: Fall back to name from patient context or cached patient data
    search_name = (patient_name or ctx.get("name") or name or "").strip()
    search_time = ctx.get("examineTime") or examine_time
    
    # If exam_date not provided, try to get from patient data
    if not exam_date and p:
        exam_date = getattr(p, "examine_time", None)
    
    logger.info(f"Building context placeholders - ris_exam_id: {ris_exam_id}, patient_name param: {patient_name}, search_name: {search_name}")

    def _merge_consultation(base: Optional[Dict[str, Any]], refined: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        keys_to_preserve = [
            "name",
            "age",
            "gender",
            "phone",
            "affectedArea",
            "leftEye",
            "rightEye",
            "bothEyes",
            "submissionTime",
            "examineTime",
        ]
        merged: Dict[str, Any] = {}
        if isinstance(base, dict):
            for key in keys_to_preserve:
                if base.get(key) is not None:
                    merged[key] = base.get(key)
        if isinstance(refined, dict):
            merged.update(refined)
        if search_name and not merged.get("name"):
            merged["name"] = search_name
        return merged

    with consultation_data_lock:
        cons_all = load_consultation_data()
        if search_name:
            idx = _find_best_matching_index(cons_all, search_name, search_time)
            if idx is not None:
                base_entry = cons_all[idx]
                refined = base_entry.get("refined") if isinstance(base_entry, dict) else None
                cons = _merge_consultation(base_entry, refined)

    if isinstance(cons, dict):
        if not age and cons.get("age"):
            age = cons.get("age")
        if not gender and cons.get("gender"):
            gender = cons.get("gender")
        if not name and cons.get("name"):
            name = cons.get("name")
        if not examine_time:
            examine_time = cons.get("examineTime") or cons.get("submissionTime")

    if (not examine_time) and isinstance(base_entry, dict):
        examine_time = (
            base_entry.get("examineTime")
            or base_entry.get("submissionTime")
            or examine_time
        )

    gender_zh = _zh_gender(gender)

    consultation_summary = _summarize_consultation(cons or {})
    # 使用改进的AI预测总结
    ai_left_summary = _summarize_ai_detailed(p, "left_eye")
    ai_right_summary = _summarize_ai_detailed(p, "right_eye")
    threshold_summary = _summarize_thresholds(p)
    manual_summary = _summarize_manual(ris_exam_id or "", exam_date)

    return {
        "patient_name": name or "患者",
        "age": str(age or ""),
        "gender_zh": gender_zh or "",
        "examine_time": examine_time or "",
        "consultation_summary": consultation_summary,
        "ai_left_summary": ai_left_summary,
        "ai_right_summary": ai_right_summary,
        "manual_summary": manual_summary,
        "threshold_summary": threshold_summary,
    }

def _fill_template(template: str, mapping: Dict[str, str]) -> str:
    # Simple placeholder replace without raising on missing keys
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v or "")
    return out


# --- LLM chat (streaming) ---
class LLMChatRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None  # Add patient_name parameter for consultation matching
    model_id: Optional[str] = None
    messages: List[Dict[str, str]]  # [{role:'user'|'assistant'|'system', content:str}]


def _llm_env():
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    base = os.getenv("LLM_API_BASE")
    if not base:
        # Keep previous local default while allowing remote API override
        base = "http://10.138.6.3:50201" if provider == "ollama" else "https://api.openai.com/v1"

    endpoint = os.getenv("LLM_CHAT_ENDPOINT")
    if not endpoint:
        endpoint = "/api/chat" if provider == "ollama" else "/v1/chat/completions"

    model_default = "DeepSeek-3.1:latest" if provider == "ollama" else "gpt-4o-mini"

    temperature_value: Optional[float] = None
    temperature_raw = os.getenv("LLM_TEMPERATURE")
    if temperature_raw is not None:
        try:
            temperature_value = float(temperature_raw)
        except ValueError:
            logger.warning("Invalid LLM_TEMPERATURE value '%s'; ignoring", temperature_raw)

    return {
        "base": base.rstrip("/"),
        "model": os.getenv("LLM_MODEL", model_default),
        "provider": provider,
        "api_key": os.getenv("LLM_API_KEY"),
        "chat_endpoint": endpoint,
        "temperature": temperature_value,
    }

@app.post("/api/llm_chat_stream")
async def llm_chat_stream(req: LLMChatRequest):
    env = _llm_env()
    cfg = load_llm_prompts()
    
    # 构建消息列表
    messages_to_send: List[Dict[str, str]] = []
    # Use original system prompt from config, but keep simple fallback
    sys_prompt = cfg.get("system_prompt", "") or "你是一个有用的AI助手。"
    
    # 检查是否是首次更新推理请求（只有一个user消息且内容是update_prompt）

    # 把req.messages都打印出来看
    logger.info(req.messages)

    is_initial_update = (
        len(req.messages or []) == 1
    )

    context_text = ""
    if cfg.get("include_patient_context", True):
        placeholders = _build_context_placeholders(req.patient_id, req.patient_name, model_id=req.model_id)
        context_text = _fill_template(cfg.get("context_template", ""), placeholders)

        if is_initial_update:
            logger.info(f"Initial context text length: {len(context_text)} characters")
        else:
            logger.info(f"Follow-up context text length: {len(context_text)} characters")

        if len(context_text) > 1000:
            context_text = context_text[:1000] + "...[TRUNCATED FOR TESTING]"
            logger.info(f"Truncated context to {len(context_text)} characters")

    if sys_prompt:
        messages_to_send.append({"role": "system", "content": sys_prompt})

    if context_text.strip():
        if len(context_text) > 5000:
            context_text = context_text[:5000] + "...[内容过长已截断]"
            logger.info(f"Trimmed context to {len(context_text)} characters before sending")
        messages_to_send.append({"role": "user", "content": context_text + "</think>"})

    if not is_initial_update and req.messages:

        req.messages = req.messages[1:]

        for msg in req.messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")
            content = msg.get("content")

            if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                logger.warning(
                    "Invalid message format: role='%s', content type=%s, content length=%s",
                    role,
                    type(content),
                    len(str(content or ""))
                )
                continue

            cleaned = content.strip()
            if role == "user":
                if not cleaned.endswith("</think>"):
                    cleaned = cleaned + "请回答这个问题。</think>"
                messages_to_send.append({"role": "user", "content": cleaned})
            elif role == "assistant":
                messages_to_send.append({"role": "assistant", "content": cleaned})
            elif role == "system":
                # Rare but allow explicit system injections from client
                messages_to_send.append({"role": "system", "content": cleaned})
            else:
                logger.warning("Unsupported message role '%s' ignored", role)

    # 验证消息列表不为空
    if not messages_to_send:
        logger.error("No valid messages to send to LLM provider")
        async def empty_response():
            yield "\n[错误] 没有有效的消息可发送\n"
        return StreamingResponse(empty_response(), media_type="text/plain; charset=utf-8")

    # === 添加DEBUG日志 ===
    logger.info("=" * 80)
    logger.info(f"LLM Chat Request for Patient: {req.patient_id} ({'INITIAL_UPDATE' if is_initial_update else 'FOLLOW_UP'})")
    logger.info("=" * 80)
    logger.info(f"Environment: {env}")
    logger.info(f"Messages count: {len(messages_to_send)}")
    
    for i, msg in enumerate(messages_to_send):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        logger.info(f"Message [{i+1}] ({role}):")
        logger.info("-" * 40)
        if len(content) > 2000:
            logger.info(f"{content[:2000]}...\n[TRUNCATED - Total length: {len(content)} chars]")
        else:
            logger.info(content)
        logger.info("-" * 40)
    
    logger.info("=" * 80)

    def _build_chat_url() -> str:
        endpoint = (env.get("chat_endpoint") or "").strip()
        base = (env.get("base") or "").rstrip("/")
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if base and endpoint:
            endpoint_clean = endpoint.lstrip("/")
            if base.endswith(endpoint_clean):
                return base
            return f"{base}/{endpoint_clean}"
        return base or endpoint

    # Use curl-like approach with simpler, more stable HTTP client configuration
    # Increase timeouts and use connection pooling similar to curl's behavior
    http_timeout = httpx.Timeout(
        connect=30.0,  # Connection timeout
        read=300.0,    # Read timeout (5 minutes for long responses)
        write=30.0,    # Write timeout
        pool=10.0      # Pool timeout
    )

    # Configure limits similar to curl's default behavior
    http_limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0
    )

    # 缓存助手回复内容
    assistant_buf: List[str] = []

    async def ollama_stream():
        url = _build_chat_url()
        payload = {
            "model": env["model"],
            "messages": messages_to_send,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE
        }

        logger.info(f"Sending request to OLLAMA: {url}")
        logger.info(f"Payload model: {payload['model']}")
        logger.info(f"Payload stream: {payload['stream']}")
        logger.info(f"Request type: {'COMBINED_USER_PROMPT' if is_initial_update else 'USER_CONTEXT_PROMPT'}")

        last_exception: Optional[Exception] = None

        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            logger.info(
                "Starting OLLAMA streaming attempt %s/%s for patient %s",
                attempt,
                OLLAMA_MAX_RETRIES,
                req.patient_id
            )

            response_chunks = 0
            pending_text = ""

            try:
                # Create client with curl-like configuration
                async with httpx.AsyncClient(
                    timeout=http_timeout,
                    limits=http_limits,
                    http2=False,  # Disable HTTP/2 for better stability
                    follow_redirects=True
                ) as client:
                    # Use simpler iter_raw approach instead of iter_lines for better reliability
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as r:
                        logger.info(f"OLLAMA Response Status: {r.status_code}")

                        if r.status_code != 200:
                            error_text = await r.aread()
                            error_msg = error_text.decode('utf-8', errors='replace')
                            last_exception = RuntimeError(f"HTTP {r.status_code}: {error_msg}")
                            logger.error("OLLAMA Error Response (attempt %s): %s", attempt, error_msg)
                            continue

                        # Process response line by line like curl does
                        buffer = b""
                        async for chunk in r.aiter_bytes(chunk_size=1024):
                            if not chunk:
                                continue

                            buffer += chunk

                            # Process complete lines
                            while b'\n' in buffer:
                                line_bytes, buffer = buffer.split(b'\n', 1)

                                try:
                                    line = line_bytes.decode('utf-8', errors='replace').strip()
                                except Exception as e:
                                    logger.warning(f"Failed to decode line: {e}")
                                    continue

                                if not line:
                                    continue

                                # Skip SSE prefixes
                                if line.startswith(":") or line.startswith("event:"):
                                    continue

                                if line.startswith("data:"):
                                    line = line[5:].strip()

                                if line in ("[DONE]", '"[DONE]"'):
                                    logger.info("OLLAMA stream signaled completion token [DONE].")
                                    break

                                # Try to parse JSON
                                try:
                                    obj = json.loads(line)
                                except json.JSONDecodeError:
                                    # Could be partial JSON, append to pending
                                    pending_text += line
                                    try:
                                        obj = json.loads(pending_text)
                                        pending_text = ""
                                    except json.JSONDecodeError:
                                        # Still not valid, wait for more data
                                        if len(pending_text) > 32768:
                                            logger.warning("Pending buffer too large, resetting")
                                            pending_text = ""
                                        continue

                                if not isinstance(obj, dict):
                                    continue

                                response_chunks += 1
                                if response_chunks == 1:
                                    logger.info("Started receiving OLLAMA streaming response...")

                                # Check for errors
                                if "error" in obj:
                                    err_text = obj.get("error")
                                    last_exception = RuntimeError(err_text)
                                    logger.error("OLLAMA Stream Error (attempt %s): %s", attempt, err_text)
                                    break

                                # Extract content
                                message = obj.get("message", {})
                                content = message.get("content", "")

                                if content:
                                    assistant_buf.append(content)
                                    yield content

                                # Check if done
                                if obj.get("done", False):
                                    logger.info(
                                        "OLLAMA response completed. Total chunks: %s, response length: %s chars",
                                        response_chunks,
                                        len(''.join(assistant_buf))
                                    )
                                    return  # Success, exit immediately

                        # Process any remaining buffer
                        if buffer:
                            try:
                                line = buffer.decode('utf-8', errors='replace').strip()
                                if line:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict) and obj.get("done"):
                                        logger.info("Stream completed with final buffer")
                                        return
                            except Exception:
                                pass

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    "OLLAMA request timeout on attempt %s/%s: %s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(
                    "OLLAMA connection error on attempt %s/%s: %s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    "OLLAMA request error on attempt %s/%s: %s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except Exception as e:
                last_exception = e
                logger.error(
                    "OLLAMA request failed on attempt %s/%s: %s",
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e),
                    exc_info=True
                )

            if assistant_buf:
                logger.info("OLLAMA streaming succeeded on attempt %s", attempt)
                return

            if attempt < OLLAMA_MAX_RETRIES:
                logger.info(
                    "Retrying OLLAMA stream (next attempt %s/%s) after %.2f seconds",
                    attempt + 1,
                    OLLAMA_MAX_RETRIES,
                    OLLAMA_RETRY_DELAY_SECONDS
                )
                await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)

        # All attempts failed
        if not assistant_buf:
            if last_exception:
                logger.error(
                    "OLLAMA streaming failed after %s attempts: %s",
                    OLLAMA_MAX_RETRIES,
                    str(last_exception)
                )
                yield f"\n[OLLAMA **错误**] {str(last_exception)}\n"
            else:
                logger.warning(
                    "OLLAMA streaming produced no content after %s attempts without explicit exception",
                    OLLAMA_MAX_RETRIES
                )
                yield "\n[LLM warning] 模型未返回内容，请稍后重试或检查日志。\n"
            return

    async def api_stream():
        url = _build_chat_url()
        payload = {
            "model": env["model"],
            "messages": messages_to_send,
            "stream": True,
        }
        if env.get("temperature") is not None:
            payload["temperature"] = env["temperature"]

        headers = {"Content-Type": "application/json"}
        api_key = (env.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        provider_name = env.get("provider", "api")
        last_exception: Optional[Exception] = None

        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            logger.info(
                "Starting %s streaming attempt %s/%s for patient %s",
                provider_name.upper(),
                attempt,
                OLLAMA_MAX_RETRIES,
                req.patient_id
            )

            response_chunks = 0

            try:
                async with httpx.AsyncClient(
                    timeout=http_timeout,
                    limits=http_limits,
                    http2=True,
                    follow_redirects=True
                ) as client:
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers
                    ) as r:
                        logger.info("%s Response Status: %s", provider_name.upper(), r.status_code)

                        if r.status_code != 200:
                            error_text = await r.aread()
                            error_msg = error_text.decode('utf-8', errors='replace')
                            last_exception = RuntimeError(f"HTTP {r.status_code}: {error_msg}")
                            logger.error("%s Error Response (attempt %s): %s", provider_name.upper(), attempt, error_msg)
                            continue

                        async for raw_line in r.aiter_lines():
                            if not raw_line:
                                continue

                            line = raw_line.strip()
                            if not line:
                                continue

                            if line.startswith(":") or line.startswith("event:"):
                                continue

                            if line.startswith("data:"):
                                line = line[5:].strip()

                            if not line:
                                continue

                            if line in ("[DONE]", '"[DONE]"'):
                                logger.info("%s stream signaled completion token [DONE].", provider_name.upper())
                                break

                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                logger.debug("Skipping non-JSON line from %s: %s", provider_name, line[:200])
                                continue

                            if isinstance(obj, dict) and obj.get("error"):
                                err_payload = obj.get("error")
                                err_msg = err_payload.get("message") if isinstance(err_payload, dict) else str(err_payload)
                                last_exception = RuntimeError(err_msg)
                                logger.error("%s Stream Error (attempt %s): %s", provider_name.upper(), attempt, err_msg)
                                break

                            choices = obj.get("choices") or []
                            chunk_text = ""

                            for choice in choices:
                                delta = choice.get("delta") or {}
                                if isinstance(delta, dict):
                                    chunk = delta.get("content")
                                    if chunk:
                                        chunk_text += chunk
                                message = choice.get("message") or {}
                                if not chunk_text and isinstance(message, dict):
                                    chunk = message.get("content")
                                    if chunk:
                                        chunk_text += chunk

                            if chunk_text:
                                response_chunks += 1
                                if response_chunks == 1:
                                    logger.info("Started receiving %s streaming response...", provider_name.upper())
                                assistant_buf.append(chunk_text)
                                yield chunk_text

                        if assistant_buf:
                            logger.info(
                                "%s response completed. Total chunks: %s, response length: %s chars",
                                provider_name.upper(),
                                response_chunks,
                                len(''.join(assistant_buf))
                            )
                            return

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    "%s request timeout on attempt %s/%s: %s",
                    provider_name.upper(),
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(
                    "%s connection error on attempt %s/%s: %s",
                    provider_name.upper(),
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    "%s request error on attempt %s/%s: %s",
                    provider_name.upper(),
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e)
                )
            except Exception as e:
                last_exception = e
                logger.error(
                    "%s request failed on attempt %s/%s: %s",
                    provider_name.upper(),
                    attempt,
                    OLLAMA_MAX_RETRIES,
                    str(e),
                    exc_info=True
                )

            if assistant_buf:
                logger.info("%s streaming succeeded on attempt %s", provider_name.upper(), attempt)
                return

            if attempt < OLLAMA_MAX_RETRIES:
                logger.info(
                    "Retrying %s stream (next attempt %s/%s) after %.2f seconds",
                    provider_name.upper(),
                    attempt + 1,
                    OLLAMA_MAX_RETRIES,
                    OLLAMA_RETRY_DELAY_SECONDS
                )
                await asyncio.sleep(OLLAMA_RETRY_DELAY_SECONDS)

        if not assistant_buf:
            if last_exception:
                logger.error(
                    "%s streaming failed after %s attempts: %s",
                    provider_name.upper(),
                    OLLAMA_MAX_RETRIES,
                    str(last_exception)
                )
                yield f"\n[LLM API **错误**] {str(last_exception)}\n"
            else:
                logger.warning(
                    "%s streaming produced no content after %s attempts without explicit exception",
                    provider_name.upper(),
                    OLLAMA_MAX_RETRIES
                )
                yield "\n[LLM warning] 模型未返回内容，请稍后重试或检查日志。\n"
            return

    stream_provider = env.get("provider", "ollama").lower()
    if stream_provider == "ollama":
        stream_fn = ollama_stream
    else:
        stream_fn = api_stream

    return StreamingResponse(stream_fn(), media_type="text/plain; charset=utf-8")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
