import io
import json
import random
from base64 import b64encode
from typing import List
import time
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from datatype import PatientData, ImageInfo, EyePrediction, EyePredictionThresholds, EyeDiagnosis, CATARACT_EXTERNAL_THRESHOLD


def _parse_ts_from_path(p: str) -> int:
    """Extract an integer timestamp-like value from filename (best-effort).
    Example: '10665476_20240926103221697.jpg' -> 20240926103221697
    Returns 0 if not found.
    """
    try:
        base = os.path.basename(p)
        name, _ = os.path.splitext(base)
        # pick the last underscore segment if any
        parts = name.split('_')
        cand = parts[-1] if parts else name
        # keep only digits
        digits = ''.join(ch for ch in cand if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0


def _pick_latest_by_type(images_json_list, image_type_mapping, desired_type: str):
    """From raw JSON images, pick the one with mapped type == desired_type and latest timestamp from img_path.
    Return the raw image dict or None.
    """
    candidates = [
        im for im in images_json_list
        if image_type_mapping.get(im.get("eye", ""), "") == desired_type
    ]
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
def load_single_patient_data(data_path: str, patient_id: str) -> PatientData:
    """
    Load and parse a single patient's data from a JSON file by patient_id.
    Returns a PatientData object or raises KeyError if not found.
    Optimized to only process the requested patient's data.
    """
    image_type_mapping = {
        "右眼眼底": "右眼CFP",
        "左眼眼底": "左眼CFP",
        "右眼外观": "右眼外眼照",
        "左眼外观": "左眼外眼照",
    }
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
    prediction_thresholds = EyePredictionThresholds()

    # Load only the JSON data first
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # Check if patient exists before processing
    if patient_id not in all_data:
        raise KeyError(f"Patient {patient_id} not found in {data_path}")
    
    pdata = all_data[patient_id]
    
    # Extract name and examine_time if available
    name = pdata.get("name")
    examine_time = pdata.get("examineTime")
    
    # Process only this patient's images
    eye_images = []
    for img in pdata.get("images", []):
        img_type = image_type_mapping.get(img.get("eye", ""), "")
        img_quality = default_image_quality
        prefix = os.path.dirname(data_path)
        img_rel = img.get("img_path", "")
        full_path = os.path.join(prefix, img_rel)
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
        cfp_img = _pick_latest_by_type(pdata.get("images", []), image_type_mapping, cfp_type)
        ext_img = _pick_latest_by_type(pdata.get("images", []), image_type_mapping, ext_type)

        if cfp_img and cfp_img.get("probs"):
            cfp_probs = {diagnosis_mapping.get(k, k): v for k, v in cfp_img.get("probs", {}).items()}
        else:
            cfp_probs = {disease: 0.0 for disease in diagnosis_mapping.values()}

        ext_probs = None
        if ext_img and ext_img.get("probs"):
            ext_probs = {diagnosis_mapping.get(k, k): v for k, v in ext_img.get("probs", {}).items()}

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
        diagnosis_results=diagnosis_results
    )

def load_batch_patient_data(data_path="../data/inference_results.json") -> List[PatientData]:
    """
    Load and parse a batch of patient data from a JSON file.
    Returns a list of PatientData objects.
    """
    image_type_mapping = {
        "右眼眼底": "右眼CFP",
        "左眼眼底": "左眼CFP",
        "右眼外观": "右眼外眼照",
        "左眼外观": "左眼外眼照",
    }
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
        # Extract name and examine_time if available
        name = pdata.get("name")
        examine_time = pdata.get("examineTime")
        
        prediction_thresholds = EyePredictionThresholds()
        # Images
        eye_images = []
        for img in pdata.get("images", []):
            img_type = image_type_mapping.get(img.get("eye", ""), "")
            img_quality = default_image_quality
            prefix = os.path.dirname(data_path)
            img_rel = img.get("img_path", "")
            full_path = os.path.join(prefix, img_rel)
            base64_data = read_image(full_path)
            eye_images.append(ImageInfo(
                id=f"img_{pid}_{img.get('img_path', '')}",
                type=img_type,
                quality=img_quality,
                base64_data=base64_data
            ))

        # Prediction results
        prediction_results = {}
        # Track whether external-eye cataract probability is used per eye
        ext_cataract_used = {"left_eye": False, "right_eye": False}
        for eye in ["left_eye", "right_eye"]:
            # Find latest CFP and 外眼 images for this eye
            cfp_type = "左眼CFP" if eye == "left_eye" else "右眼CFP"
            ext_type = "左眼外眼照" if eye == "left_eye" else "右眼外眼照"
            cfp_img = _pick_latest_by_type(pdata.get("images", []), image_type_mapping, cfp_type)
            ext_img = _pick_latest_by_type(pdata.get("images", []), image_type_mapping, ext_type)

            # Base probabilities from CFP (or zeros if missing)
            if cfp_img and cfp_img.get("probs"):
                cfp_probs = {diagnosis_mapping.get(k, k): v for k, v in cfp_img.get("probs", {}).items()}
            else:
                cfp_probs = {disease: 0.0 for disease in diagnosis_mapping.values()}

            # External-eye probabilities (if available)
            ext_probs = None
            if ext_img and ext_img.get("probs"):
                ext_probs = {diagnosis_mapping.get(k, k): v for k, v in ext_img.get("probs", {}).items()}

            # Start with CFP-based probs
            probs = dict(cfp_probs)
            # Prefer 外眼 cataract if available
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
                # Use a separate threshold for cataract if external-eye prob was used
                if disease == "白内障" and ext_cataract_used.get(eye, False):
                    threshold_to_use = CATARACT_EXTERNAL_THRESHOLD
                else:
                    threshold_to_use = threshold
                prob = getattr(prediction_results[eye], disease, 0.0)
                diag[disease] = prob >= threshold_to_use
            diagnosis_results[eye] = EyeDiagnosis(**diag)

        patients.append(PatientData(
            patient_id=pid,
            name=name,  # Added name field
            examine_time=examine_time,  # Added examine_time field
            eye_images=eye_images,
            prediction_results=prediction_results,
            prediction_thresholds=prediction_thresholds,
            diagnosis_results=diagnosis_results
        ))
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
    prediction_thresholds = EyePredictionThresholds()

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
        diagnosis_results=diagnosis_results
    )

def create_batch_dummy_patient_data(num_patients: int = 5) -> List[PatientData]:
    """Creates an initial batch of dummy data with random predictions and threshold-derived diagnoses."""
    patients: List[PatientData] = []
    for _ in range(num_patients):
        patients.append(create_single_dummy_patient_data())
    return patients