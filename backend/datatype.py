from pydantic import BaseModel
from typing import List, Dict

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
    eye_images: List[ImageInfo]
    prediction_thresholds: EyePredictionThresholds
    prediction_results: Dict[str, EyePrediction] # 'left_eye', 'right_eye'
    diagnosis_results: Dict[str, EyeDiagnosis] # 'left_eye', 'right_eye'


class SubmitDiagnosisRequest(BaseModel):
    patient_id: str
    image_updates: List[Dict[str, str]] = None  # Only id, type, quality
    # Manual diagnosis data
    manual_diagnosis: Dict[str, ManualEyeDiagnosis] = None
    custom_diseases: CustomDiseases = None
    diagnosis_notes: str = ""
    
