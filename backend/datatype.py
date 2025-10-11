from pydantic import BaseModel
from typing import List, Dict, Optional

# --- Data Models (Pydantic) ---
class ImageInfo(BaseModel):
    id: str
    type: str  # i.e., '左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'
    quality: str # i.e., '图像质量高', '图像质量可用', '图像质量差'
    base64_data: str # Base64 encoded image


class EyePrediction(BaseModel):
    # Eye disease prediction scores
    青光眼: float = 0.0
    糖网: float = 0.0
    AMD: float = 0.0
    病理性近视: float = 0.0
    RVO: float = 0.0
    RAO: float = 0.0
    视网膜脱离: float = 0.0
    其它视网膜病: float = 0.0
    其它黄斑病变: float = 0.0
    白内障: float = 0.0
    正常: float = 0.0 # 'Normal'

class EyePredictionThresholds(BaseModel):
    青光眼: float = 0.6
    糖网: float = 0.5
    AMD: float = 0.3
    病理性近视: float = 0.1
    RVO: float = 0.2
    RAO: float = 0.02
    视网膜脱离: float = 0.2
    其它视网膜病: float = 0.2
    其它黄斑病变: float = 0.5
    白内障: float = 0.6
    正常: float = 0.08 # 'Normal'
    
    @classmethod
    def get_threshold_set_1(cls):
        """First candidate threshold set (optimal F1 scores)"""
        return cls(
            青光眼=0.18,      # Class 0
            糖网=0.23,        # Class 1  
            AMD=0.18,         # Class 2
            病理性近视=0.18,   # Class 3
            RVO=0.18,         # Class 4
            RAO=0.15,         # Class 5
            视网膜脱离=0.15,   # Class 6
            其它视网膜病=0.35,  # Class 7
            其它黄斑病变=0.50,  # Class 8
            白内障=0.33,      # Class 9
            正常=0.38         # Class 10
        )
    
    @classmethod
    def get_threshold_set_2(cls):
        """Second candidate threshold set (alternative F1 scores)"""
        return cls(
            青光眼=0.20,      # Class 0
            糖网=0.28,        # Class 1
            AMD=0.25,         # Class 2
            病理性近视=0.18,   # Class 3
            RVO=0.20,         # Class 4
            RAO=0.18,         # Class 5
            视网膜脱离=0.18,   # Class 6
            其它视网膜病=0.25,  # Class 7
            其它黄斑病变=0.23,  # Class 8
            白内障=0.28,      # Class 9
            正常=0.50         # Class 10
        )
    
    @classmethod
    def get_threshold_set(cls, set_index: int):
        """Get threshold set by index (0 or 1)"""
        if set_index == 0:
            return cls.get_threshold_set_1()
        elif set_index == 1:
            return cls.get_threshold_set_2()
        else:
            return cls.get_threshold_set_1()  # Default to set 1


# Separate threshold for cataract when using external-eye (外眼) predictions
CATARACT_EXTERNAL_THRESHOLD = 0.6


class EyeDiagnosis(BaseModel):
    # Dictionary where keys are disease names and values are boolean (detected/not detected)
    青光眼: bool = False
    糖网: bool = False
    AMD: bool = False
    病理性近视: bool = False
    RVO: bool = False
    RAO: bool = False
    视网膜脱离: bool = False
    其它视网膜病: bool = False
    其它黄斑病变: bool = False
    白内障: bool = False
    正常: bool = False # 'Normal'



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
    manual_diagnosis: Dict[str, ManualEyeDiagnosis]  # 'left_eye', 'right_eye'
    custom_diseases: CustomDiseases
    diagnosis_notes: str = ""


class PatientData(BaseModel):
    patient_id: str
    name: Optional[str] = None
    examine_time: Optional[str] = None
    eye_images: List[ImageInfo]
    prediction_results: Dict[str, EyePrediction]
    diagnosis_results: Dict[str, EyeDiagnosis]
    prediction_thresholds: EyePredictionThresholds
    active_threshold_set: int = 0  # 0 for set 1, 1 for set 2

class SubmitDiagnosisRequest(BaseModel):
    patient_id: str
    image_updates: Optional[List[Dict[str, str]]] = None  # Only id, type, quality
    # Manual diagnosis data - accept raw dicts for flexibility
    manual_diagnosis: Optional[Dict[str, Dict[str, bool]]] = None
    custom_diseases: Optional[CustomDiseases] = None
    diagnosis_notes: str = ""
    

class UpdateSelectionRequest(BaseModel):
    patient_id: str
    selected_image_ids: List[str]
    exam_date: Optional[str] = None  # Optional exam date (YYYYMMDD format)
    

class AlterThresholdRequest(BaseModel):
    patient_id: str
    exam_date: Optional[str] = None  # Optional exam date (YYYYMMDD format)
