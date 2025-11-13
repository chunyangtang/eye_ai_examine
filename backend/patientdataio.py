import io
import json
import random
import os
import copy
from base64 import b64encode
from typing import List, Dict, Any, Optional
import time

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from dotenv import load_dotenv

from datatype import PatientData, ImageInfo, CATARACT_EXTERNAL_THRESHOLD

# Load environment variables
load_dotenv()


def _parse_ts_from_path(p: str) -> int:
    """Extract an integer timestamp-like value from filename (best-effort).
    Example: '10665476_20240926103221697.jpg' -> 20240926103221697
    Returns 0 if not found.
    """
    try:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        # 先尝试取最后一段
        parts = name.split('_')
        cand = parts[-1] if parts else name
        digits = ''.join(ch for ch in cand if ch.isdigit())
        if digits:
            return int(digits)
        # 回退：全名里收集所有数字
        digits_all = ''.join(ch for ch in name if ch.isdigit())
        return int(digits_all) if digits_all else 0
    except Exception:
        return 0


def _map_eye_class_to_type(eye_cls: str) -> Optional[str]:
    """Map RAW v2 eye_classification.class to internal types."""
    cls = (eye_cls or "").strip().lower()
    if cls == "left fundus":
        return "左眼CFP"
    if cls == "right fundus":
        return "右眼CFP"
    if cls == "left outer eye":
        return "左眼外眼照"
    if cls == "right outer eye":
        return "右眼外眼照"
    return None


def _extract_diseases_from_img(img: Dict[str, Any]) -> Dict[str, float]:
    """Extract diseases probs from fundus_classification.diseases."""
    try:
        diseases = img.get("fundus_classification", {}).get("diseases", {})
        if isinstance(diseases, dict):
            return {str(k): float(v) for k, v in diseases.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def _normalize_probability_map(raw_probs: Optional[Dict[str, float]], alias_map: Dict[str, str], disease_keys: List[str]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not raw_probs:
        raw_probs = {}
    for raw_key, value in raw_probs.items():
        canonical = alias_map.get(raw_key)
        if not canonical:
            canonical = alias_map.get(str(raw_key).lower(), raw_key)
        normalized[canonical] = float(value)
    for disease_key in disease_keys:
        normalized.setdefault(disease_key, 0.0)
    return normalized


def _pick_latest_by_type(images_json_list: List[Dict[str, Any]], desired_type: str):
    """
    From RAW v2 images, pick the one with mapped type == desired_type and latest timestamp by img_path.
    """
    candidates = []
    for im in images_json_list or []:
        eye_cls = (im.get("eye_classification", {}) or {}).get("class", "")
        mapped = _map_eye_class_to_type(eye_cls)
        if mapped == desired_type:
            candidates.append(im)
    if not candidates:
        return None
    candidates.sort(key=lambda im: _parse_ts_from_path(im.get("img_path", "")), reverse=True)
    return candidates[0]


def read_image(image_path):
    """Reads an image from the given path and returns its Base64 encoded string."""
    if not image_path:
        return None
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


# --- Load a single patient from JSON (Optimized) ---
def _build_patient_data_from_payload(
    patient_id: str,
    pdata: Dict[str, Any],
    base_dir: str,
    model_cfg: Dict[str, Any],
    exam_date: Optional[str] = None,
) -> PatientData:
    """Transform a preloaded patient payload into a PatientData object."""
    default_image_quality = ""
    disease_entries = model_cfg.get("diseases") or []
    disease_keys = [entry.get("key") for entry in disease_entries if entry.get("key")]
    alias_map = model_cfg.get("disease_alias_map", {})
    if not disease_keys:
        # fallback to alias map keys if config missing entries
        disease_keys = sorted(set(alias_map.values()))
    threshold_sets = model_cfg.get("threshold_sets") or []
    threshold_set_indices = model_cfg.get("threshold_set_indices", {})
    default_set_id = model_cfg.get("default_threshold_set_id") or (threshold_sets[0]["id"] if threshold_sets else None)
    active_idx = threshold_set_indices.get(default_set_id, 0)
    active_thresholds = dict(threshold_sets[active_idx]["values"]) if threshold_sets else {}

    # Extract name and examine_time if available
    name = pdata.get("name")
    examine_time = pdata.get("examineTime")
    
    # Process only this patient's images
    eye_images = []
    for img in pdata.get("images", []):
        if not isinstance(img, dict):
            continue
        eye_cls = (img.get("eye_classification", {}) or {}).get("class", "")
        img_type = _map_eye_class_to_type(eye_cls) or ""
        img_quality = default_image_quality
        img_rel = img.get("img_path", "")
        # 兼容绝对/相对路径
        if os.path.isabs(img_rel):
            full_path = img_rel
        else:
            full_path = os.path.join(base_dir, img_rel)
        base64_data = read_image(full_path)
        eye_images.append(ImageInfo(
            id=f"img_{patient_id}_{img.get('img_path', '')}",
            type=img_type,
            quality=img_quality,
            base64_data=base64_data or ""
        ))

    # Process prediction results
    prediction_results: Dict[str, Dict[str, float]] = {}
    ext_cataract_used = {"left_eye": False, "right_eye": False}

    image_list = pdata.get("images", [])

    for eye in ["left_eye", "right_eye"]:
        cfp_type = "左眼CFP" if eye == "left_eye" else "右眼CFP"
        ext_type = "左眼外眼照" if eye == "left_eye" else "右眼外眼照"
        cfp_img = _pick_latest_by_type(image_list, cfp_type)
        ext_img = _pick_latest_by_type(image_list, ext_type)

        if cfp_img:
            cfp_probs = _normalize_probability_map(_extract_diseases_from_img(cfp_img), alias_map, disease_keys)
        else:
            cfp_probs = {disease: 0.0 for disease in disease_keys}

        ext_probs = {}
        if ext_img:
            ext_probs = _normalize_probability_map(_extract_diseases_from_img(ext_img), alias_map, disease_keys)

        combined = dict(cfp_probs)
        if "白内障" in ext_probs:
            combined["白内障"] = ext_probs["白内障"]
            ext_cataract_used[eye] = True

        prediction_results[eye] = {key: float(combined.get(key, 0.0)) for key in disease_keys}

    # Adjust cataract thresholds if external eye probabilities are used
    adjusted_thresholds = dict(active_thresholds)
    if (ext_cataract_used["left_eye"] or ext_cataract_used["right_eye"]) and "白内障" in adjusted_thresholds:
        adjusted_thresholds["白内障"] = CATARACT_EXTERNAL_THRESHOLD

    # Diagnosis results (thresholding)
    diagnosis_results: Dict[str, Dict[str, bool]] = {}
    for eye in ["left_eye", "right_eye"]:
        diag = {}
        for disease, threshold in adjusted_thresholds.items():
            prob = float(prediction_results.get(eye, {}).get(disease, 0.0))
            diag[disease] = prob >= float(threshold)
        diagnosis_results[eye] = diag

    return PatientData(
        patient_id=patient_id,
        name=name,
        examine_time=examine_time,
        eye_images=eye_images,
        prediction_results=prediction_results,
        prediction_thresholds=adjusted_thresholds,
        diagnosis_results=diagnosis_results,
        active_threshold_set=active_idx,
        active_threshold_set_id=threshold_sets[active_idx]["id"] if threshold_sets else None,
        active_threshold_set_index=active_idx,
        threshold_sets=copy.deepcopy(threshold_sets),
        model_id=model_cfg.get("id"),
        model_name=model_cfg.get("name"),
        diseases=model_cfg.get("diseases", []),
    )


def load_single_patient_data(data_path: str, patient_id: str, model_cfg: Dict[str, Any]) -> PatientData:
    """
    Load and parse a single patient's data from a JSON file by patient_id.
    Returns a PatientData object or raises KeyError if not found.
    Optimized to only process the requested patient's data.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if patient_id not in all_data:
        raise KeyError(f"Patient {patient_id} not found in {data_path}")

    base_dir = os.path.dirname(data_path)
    return _build_patient_data_from_payload(patient_id, all_data[patient_id], base_dir, model_cfg)


def load_patient_from_record(patient_id: str, payload: Dict[str, Any], base_dir: str, model_cfg: Dict[str, Any], exam_date: Optional[str] = None) -> PatientData:
    """Public helper to build PatientData from preloaded JSON payload and base directory."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict containing patient data")
    return _build_patient_data_from_payload(patient_id, payload, base_dir, model_cfg, exam_date=exam_date)

def load_batch_patient_data(model_cfg: Dict[str, Any], data_path=None) -> List[PatientData]:
    """
    Load and parse a batch of patient data from a JSON file.
    Returns a list of PatientData objects.
    """
    if data_path is None:
        data_path_env = os.getenv("RAW_JSON_PATH", "../data/inference_results.json")
        if not os.path.isabs(data_path_env):
            data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), data_path_env))
        else:
            data_path = data_path_env
    default_image_quality = ""
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
    # Create thresholds per patient inside the loop to avoid cross-patient mutation

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patients = []
    for pid, pdata in tqdm(data.items(), desc="Loading patient data"):
        patients.append(_build_patient_data_from_payload(pid, pdata, os.path.dirname(data_path), model_cfg))
    return patients


# --- Helper Functions of Initial Demo ---
def generate_random_image_base64(text: str, color: tuple) -> str:
    """Generates a simple colored square image with text and returns its Base64 string.
    """
    width, height = 200, 200
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    text_color = (255, 255, 255) if sum(color) < 300 else (0, 0, 0)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (width - text_width) / 2
    text_y = (height - text_height) / 2

    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode("utf-8")


DUMMY_DISEASE_KEYS = ["青光眼", "糖网", "AMD", "病理性近视", "RVO", "RAO", "视网膜脱离", "其它视网膜病", "其它黄斑病变", "白内障", "正常"]
DUMMY_THRESHOLD_SETS = [
    {
        "id": "set_1",
        "name": "示例阈值",
        "description": "用于占位的数据集",
        "values": {key: 0.5 for key in DUMMY_DISEASE_KEYS},
    }
]


def create_single_dummy_patient_data() -> PatientData:
    """Creates dummy patient data including random predictions and derived diagnoses."""
    patient_id = f"external_trigger_{int(time.time())}_{random.randint(100, 999)}"

    # --- Images ---
    eye_images: List[ImageInfo] = []
    image_types = ['左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照']
    image_qualities = ['图像质量高', '图像质量可用', '图像质量差', '']
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(6)]
    for i in range(6):  # fixed number for predictable UI density; adjust if needed
        img_type = random.choice(image_types)
        img_quality = random.choice(image_qualities)
        img_color = colors[i]
        base64_data = generate_random_image_base64(f"{img_type}\nQ:{img_quality}\nImg {i+1}", img_color)
        eye_images.append(ImageInfo(
            id=f"img_{patient_id}_{i}",
            type=img_type,
            quality=img_quality,
            base64_data=base64_data
        ))

    # --- Random Predictions ---
    def random_prediction() -> Dict[str, float]:
        probs = {}
        for key in DUMMY_DISEASE_KEYS:
            threshold_val = DUMMY_THRESHOLD_SETS[0]["values"].get(key, 0.5)
            if random.random() < 0.2:
                jitter = (random.random() - 0.5) * 0.2
                val = min(1.0, max(0.0, threshold_val + jitter))
            else:
                val = random.random()
            probs[key] = round(val, 4)
        return probs

    left_pred = random_prediction()
    right_pred = random_prediction()

    prediction_results = {
        "left_eye": left_pred,
        "right_eye": right_pred
    }

    def derive_diagnosis(probs: Dict[str, float]) -> Dict[str, bool]:
        diag_flags = {}
        for disease, threshold_val in DUMMY_THRESHOLD_SETS[0]["values"].items():
            prob_val = probs.get(disease, 0.0)
            diag_flags[disease] = prob_val >= threshold_val
        return diag_flags

    diagnosis_results = {
        "left_eye": derive_diagnosis(left_pred),
        "right_eye": derive_diagnosis(right_pred)
    }

    return PatientData(
        patient_id=patient_id,
        eye_images=eye_images,
        prediction_thresholds=DUMMY_THRESHOLD_SETS[0]["values"].copy(),
        prediction_results=prediction_results,
        diagnosis_results=diagnosis_results,
        active_threshold_set=0,
        active_threshold_set_id=DUMMY_THRESHOLD_SETS[0]["id"],
        active_threshold_set_index=0,
        threshold_sets=copy.deepcopy(DUMMY_THRESHOLD_SETS),
        model_id="dummy",
        model_name="Dummy Model",
        diseases=[{"key": key, "label_cn": key, "label_en": key, "short_name": key} for key in DUMMY_DISEASE_KEYS],
    )

def create_batch_dummy_patient_data(num_patients: int = 5) -> List[PatientData]:
    """Creates an initial batch of dummy data with random predictions and threshold-derived diagnoses."""
    patients: List[PatientData] = []
    for _ in range(num_patients):
        patients.append(create_single_dummy_patient_data())
    return patients
