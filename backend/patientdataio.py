import io
import json
import random
import os
from base64 import b64encode
from typing import List, Dict, Any, Optional
import time

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from dotenv import load_dotenv

from datatype import PatientData, ImageInfo, EyePrediction, EyePredictionThresholds, EyeDiagnosis, CATARACT_EXTERNAL_THRESHOLD

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
def _build_patient_data_from_payload(patient_id: str, pdata: Dict[str, Any], base_dir: str) -> PatientData:
    """Transform a preloaded patient payload into a PatientData object."""
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
    prediction_thresholds = EyePredictionThresholds.get_threshold_set_1()  # Default to set 1

    # Extract name and examine_time if available
    name = pdata.get("name")
    examine_time = pdata.get("examineTime")
    
    # Process only this patient's images
    eye_images = []
    for img in pdata.get("images", []):
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
            base64_data=base64_data
        ))

    # Process prediction results
    prediction_results = {}
    ext_cataract_used = {"left_eye": False, "right_eye": False}
    for eye in ["left_eye", "right_eye"]:
        cfp_type = "左眼CFP" if eye == "left_eye" else "右眼CFP"
        ext_type = "左眼外眼照" if eye == "left_eye" else "右眼外眼照"
        cfp_img = _pick_latest_by_type(pdata.get("images", []), cfp_type)
        ext_img = _pick_latest_by_type(pdata.get("images", []), ext_type)

        if cfp_img:
            cfp_probs_raw = _extract_diseases_from_img(cfp_img)
            cfp_probs = {diagnosis_mapping.get(k, k): v for k, v in cfp_probs_raw.items()}
        else:
            cfp_probs = {disease: 0.0 for disease in diagnosis_mapping.values()}

        ext_probs = None
        if ext_img:
            ext_probs_raw = _extract_diseases_from_img(ext_img)
            if ext_probs_raw:
                ext_probs = {diagnosis_mapping.get(k, k): v for k, v in ext_probs_raw.items()}

        probs = dict(cfp_probs)
        if ext_probs is not None and "白内障" in ext_probs:
            probs["白内障"] = ext_probs["白内障"]
            ext_cataract_used[eye] = True

        prediction_results[eye] = EyePrediction(**probs)

    # If either eye uses external-eye cataract probability, reflect that in thresholds for UI
    if ext_cataract_used["left_eye"] or ext_cataract_used["right_eye"]:
        setattr(prediction_thresholds, "白内障", CATARACT_EXTERNAL_THRESHOLD)

    # Diagnosis results (thresholding)
    diagnosis_results = {}
    for eye in ["left_eye", "right_eye"]:
        diag = {}
        for disease, threshold in prediction_thresholds.dict().items():
            if disease == "白内障" and ext_cataract_used.get(eye, False):
                threshold_to_use = CATARACT_EXTERNAL_THRESHOLD
            else:
                threshold_to_use = threshold
            prob = getattr(prediction_results[eye], disease, 0.0)
            diag[disease] = prob >= threshold_to_use
        diagnosis_results[eye] = EyeDiagnosis(**diag)

    return PatientData(
        patient_id=patient_id,
        name=name,  # Added name field
        examine_time=examine_time,  # Added examine_time field
        eye_images=eye_images,
        prediction_results=prediction_results,
        prediction_thresholds=prediction_thresholds,
        diagnosis_results=diagnosis_results,
        active_threshold_set=0  # Default to threshold set 1
    )


def load_single_patient_data(data_path: str, patient_id: str) -> PatientData:
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
    return _build_patient_data_from_payload(patient_id, all_data[patient_id], base_dir)


def load_patient_from_record(patient_id: str, payload: Dict[str, Any], base_dir: str) -> PatientData:
    """Public helper to build PatientData from preloaded JSON payload and base directory."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict containing patient data")
    return _build_patient_data_from_payload(patient_id, payload, base_dir)

def load_batch_patient_data(data_path=None) -> List[PatientData]:
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
        patients.append(_build_patient_data_from_payload(pid, pdata, os.path.dirname(data_path)))
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

    # --- Prediction Thresholds ---
    prediction_thresholds = EyePredictionThresholds.get_threshold_set_1()  # Default to set 1

    # --- Random Predictions ---
    def random_prediction() -> EyePrediction:
        # Bias probabilities around thresholds: sample from uniform but occasionally spike
        probs = {}
        for field_name in EyePrediction.__fields__.keys():
            base = random.random()
            # 20% chance to concentrate near threshold region (±0.1) to exercise UI at decision boundary
            threshold_val = getattr(prediction_thresholds, field_name, 0.5)
            if random.random() < 0.2:
                jitter = (random.random() - 0.5) * 0.2  # ±0.1
                val = min(1.0, max(0.0, threshold_val + jitter))
            else:
                val = base
            probs[field_name] = round(val, 4)
        return EyePrediction(**probs)

    left_pred = random_prediction()
    right_pred = random_prediction()

    prediction_results = {
        "left_eye": left_pred,
        "right_eye": right_pred
    }

    # --- Derive Diagnosis from Predictions & Thresholds ---
    def derive_diagnosis(pred: EyePrediction) -> EyeDiagnosis:
        diag_flags = {}
        for disease in EyeDiagnosis.__fields__.keys():
            threshold_val = getattr(prediction_thresholds, disease, 0.5)
            prob_val = getattr(pred, disease, 0.0)
            diag_flags[disease] = prob_val >= threshold_val
        return EyeDiagnosis(**diag_flags)

    diagnosis_results = {
        "left_eye": derive_diagnosis(left_pred),
        "right_eye": derive_diagnosis(right_pred)
    }

    return PatientData(
        patient_id=patient_id,
        eye_images=eye_images,
        prediction_thresholds=prediction_thresholds,
        prediction_results=prediction_results,
        diagnosis_results=diagnosis_results,
        active_threshold_set=0  # Default to threshold set 1
    )

def create_batch_dummy_patient_data(num_patients: int = 5) -> List[PatientData]:
    """Creates an initial batch of dummy data with random predictions and threshold-derived diagnoses."""
    patients: List[PatientData] = []
    for _ in range(num_patients):
        patients.append(create_single_dummy_patient_data())
    return patients