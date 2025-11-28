from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# --- Data Models (Pydantic) ---
class ImageInfo(BaseModel):
    id: str
    type: str  # i.e., '左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'
    quality: str # i.e., '图像质量高', '图像质量可用', '图像质量差'
    base64_data: str # Base64 encoded image


# Separate threshold for cataract when using external-eye (外眼) predictions
CATARACT_EXTERNAL_THRESHOLD = 0.6

class ManualEyeDiagnosis(BaseModel):
    # Manual diagnosis for diseases not tied to AI predictions
    青光眼: bool = False
    糖网: bool = False
    AMD: bool = False
    病理性近视: bool = False
    高度近视: bool = False  # High myopia - new addition
    RVO: bool = False
    RAO: bool = False
    视网膜脱离: bool = False
    其它视网膜病: bool = False
    其它黄斑病变: bool = False
    白内障: bool = False
    正常: bool = False

class CustomDiseases(BaseModel):
    left_eye: str = ""
    right_eye: str = ""

class ManualDiagnosisData(BaseModel):
    manual_diagnosis: Dict[str, Dict[str, bool]]  # 'left_eye', 'right_eye'
    custom_diseases: CustomDiseases
    diagnosis_notes: str = ""


class PatientData(BaseModel):
    patient_id: str
    name: Optional[str] = None
    examine_time: Optional[str] = None
    eye_images: List[ImageInfo]
    prediction_results: Dict[str, Dict[str, float]]
    diagnosis_results: Dict[str, Dict[str, bool]]
    prediction_thresholds: Dict[str, float]
    active_threshold_set: int = 0  # kept for backward compatibility
    active_threshold_set_id: Optional[str] = None
    active_threshold_set_index: int = 0
    threshold_sets: List[Dict[str, Any]] = []
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    diseases: List[Dict[str, Any]] = []
    disease_alias_map: Dict[str, str] = {}

class SubmitDiagnosisRequest(BaseModel):
    patient_id: str
    exam_date: Optional[str] = None  # Optional exam date (YYYYMMDD format)
    image_updates: Optional[List[Dict[str, str]]] = None  # Only id, type, quality
    # Manual diagnosis data - accept raw dicts for flexibility
    manual_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None
    custom_diseases: Optional[CustomDiseases] = None
    diagnosis_notes: str = ""
    model_id: Optional[str] = None
    

class UpdateSelectionRequest(BaseModel):
    patient_id: str
    selected_image_ids: List[Optional[str]]
    exam_date: Optional[str] = None  # Optional exam date (YYYYMMDD format)
    model_id: Optional[str] = None
    

class AlterThresholdRequest(BaseModel):
    patient_id: str
    exam_date: Optional[str] = None  # Optional exam date (YYYYMMDD format)
    model_id: Optional[str] = None
    threshold_set_id: Optional[str] = None
