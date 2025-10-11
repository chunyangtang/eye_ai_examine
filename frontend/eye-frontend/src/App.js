import React, { useState, useEffect, useCallback, useRef, useMemo, useLayoutEffect } from 'react';

const renderMarkdownContent = (content) => {
  if (!content) return '';
  let html = String(content).replace(/\r\n/g, '\n');

  // Headings H1–H6
  html = html.replace(/^###### (.*?)(\n|$)/gm, '<h6 class="text-xs font-semibold text-gray-700 mt-1 mb-1">$1</h6>$2');
  html = html.replace(/^##### (.*?)(\n|$)/gm, '<h5 class="text-sm font-semibold text-gray-700 mt-2 mb-1">$1</h5>$2');
  html = html.replace(/^#### (.*?)(\n|$)/gm, '<h4 class="text-base font-semibold text-gray-800 mt-3 mb-1">$1</h4>$2');
  html = html.replace(/^### (.*?)(\n|$)/gm, '<h3 class="text-lg font-semibold text-gray-800 mt-3 mb-2">$1</h3>$2');
  html = html.replace(/^## (.*?)(\n|$)/gm, '<h2 class="text-xl font-bold text-gray-900 mt-4 mb-2">$1</h2>$2');
  html = html.replace(/^# (.*?)(\n|$)/gm, '<h1 class="text-2xl font-extrabold text-gray-900 mt-5 mb-3">$1</h1>$2');

  // Group unordered lists (-, *, +)
  html = html.replace(/(^|\n)((?:\s*[-*+]\s+.+\n?)+)/g, (m, p1, block) => {
    const items = block
      .trimEnd()
      .split('\n')
      .map(line => line.replace(/^\s*[-*+]\s+(.+)$/, '<li class="mb-1">$1</li>'))
      .join('');
    return `${p1}<ul class="list-disc pl-5 my-2">${items}</ul>\n`;
  });

  // Group ordered lists (1. 2. ...)
  html = html.replace(/(^|\n)((?:\s*\d+\.\s+.+\n?)+)/g, (m, p1, block) => {
    const items = block
      .trimEnd()
      .split('\n')
      .map(line => line.replace(/^\s*\d+\.\s+(.+)$/, '<li class="mb-1">$1</li>'))
      .join('');
    return `${p1}<ol class="list-decimal pl-5 my-2">${items}</ol>\n`;
  });

  // Bold/Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
  html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em class="italic">$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1 py-0.5 rounded text-[85%]">$1</code>');

  // Blockquote
  html = html.replace(/(^|\n)>\s?(.*)(?=\n|$)/g, '$1<blockquote class="border-l-4 border-gray-300 pl-3 italic text-gray-700 my-2">$2</blockquote>');

  // Horizontal rule
  html = html.replace(/^\s*([-*_]){3,}\s*$/gm, '<hr class="my-3 border-t border-gray-200"/>');

  // Remaining line breaks
  html = html.replace(/\n/g, '<br>');

  return html;
};

// Reusable Button Component
const IconButton = ({ children, onClick, className = '' }) => (
  <button
    onClick={onClick}
    className={`p-2 rounded-full shadow-md transition-all duration-200 ease-in-out
    bg-white text-gray-700 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500
    ${className}`}
  >
    {children}
  </button>
);

// Reselect Image Modal Component
const ReselectImageModal = ({ isOpen, onClose, allImages, onSelectImages, selectedImageIds }) => {
  const [currentSelection, setCurrentSelection] = useState([...selectedImageIds]);

  useEffect(() => {
    setCurrentSelection([...selectedImageIds]);
  }, [selectedImageIds, isOpen]);

  const handleImageClick = (imageId, slotIndex) => {
    const newSelection = [...currentSelection];
    const imageAlreadySelectedInAnotherSlot = newSelection.findIndex((id, index) => id === imageId && index !== slotIndex);

    // If the image is already selected in another slot, swap them
    if (imageAlreadySelectedInAnotherSlot !== -1) {
      const oldImageIdInTargetSlot = newSelection[slotIndex];
      newSelection[slotIndex] = imageId;
      newSelection[imageAlreadySelectedInAnotherSlot] = oldImageIdInTargetSlot;
    } else {
      // Otherwise, just update the selected slot
      newSelection[slotIndex] = imageId;
    }
    setCurrentSelection(newSelection);
  };

  const handleSubmit = () => {
    onSelectImages(currentSelection);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-75 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-xl w-11/12 max-w-2xl max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">重新选择图片 (Reselect Images)</h2>
        <p className="text-sm text-gray-600 mb-4">Click an image below, then click one of the 4 slots to assign it. Click the same slot to deselect. You can also drag images to swap slots.</p>

        {currentSelection.map((selectedId, slotIndex) => (
          <div key={slotIndex} className="mb-4">
            <h3 className="text-md font-semibold mb-2">Slot {slotIndex + 1} ({allImages.find(img => img.id === selectedId)?.type || 'Empty'})</h3>
            <div className="flex flex-wrap gap-2 justify-center border p-2 rounded-md bg-gray-50 min-h-[100px]">
              {allImages.map(image => (
                <div
                  key={image.id}
                  className={`relative cursor-pointer border-2 rounded-md overflow-hidden
                    ${currentSelection[slotIndex] === image.id ? 'border-blue-500 ring-2 ring-blue-500' : 'border-transparent'}
                    ${currentSelection.includes(image.id) && currentSelection[slotIndex] !== image.id ? 'opacity-50' : ''}
                    hover:border-blue-300 transition-all duration-150`}
                  onClick={() => handleImageClick(image.id, slotIndex)}
                >
                  <img src={`data:image/png;base64,${image.base64_data}`} alt={image.type} className="w-24 h-24 object-cover" />
                  <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-1 text-center truncate">
                    {image.type}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 transition-colors"
          >
            取消 (Cancel)
          </button>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            确认 (Confirm)
          </button>
        </div>
      </div>
    </div>
  );
};

// New Expanded Image Modal Component
const ExpandedImageModal = ({ isOpen, onClose, imageInfo }) => {
  if (!isOpen || !imageInfo) return null;

  return (
    <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex justify-center items-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-[90vmin] h-[90vmin] flex flex-col">
        <div className="flex justify-between items-center p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-gray-800">{imageInfo.type} - {imageInfo.quality}</h2>
          <IconButton onClick={onClose} className="bg-transparent hover:bg-gray-100 p-1">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </IconButton>
        </div>
        <div className="p-4 flex-grow flex justify-center items-center overflow-auto">
          <img
            src={`data:image/png;base64,${imageInfo.base64_data}`}
            alt={imageInfo.type}
            className="w-full h-full object-contain"
            style={{ maxWidth: '100%', maxHeight: '100%', width: '100%', height: '100%', display: 'block', margin: 'auto' }}
          />
        </div>
      </div>
    </div>
  );
};

// ConsultationInfoSection component - This will be displayed on the left
const ConsultationInfoSection = ({ consultationData, onChange, onSubmit, isSubmitting }) => {
  // Mapping of choice codes to human-readable text
  const symptomMapping = {
    "A": "视物模糊/视力下降",
    "B": "眼前出现漂浮物、黑影飘动",
    "C": "闪光感/水波纹/视物遮挡", 
    "D": "眼红/充血/分泌物增多/眼痒",
    "E": "眼睛干涩/异物感/疲劳",
    "F": "眼睛长了小疙瘩/肿物（眼睑/角膜/结膜）",
    "G": "眼痛/眼眶痛",
    "H": "眼球突出",
    "I": "其他症状"
  };
  
  const onsetMapping = {
    "A": "突发性（数小时内迅速出现）",
    "B": "渐进性（数天或更长时间缓慢加重）"
  };
  
  const historyMapping = {
    "A": "近期外伤、眼部手术或接触化学物质",
    "B": "有高血压、糖尿病等全身疾病史",
    "C": "长期戴隐形眼镜或屈光不正史",
    "D": "长期屏幕使用史",
    "E": "无明显诱因或既往病史"
  };
  
  // Format accompanying symptoms
  const formatAccompanyingSymptoms = (symptoms) => {
    if (!symptoms) return "";
    if (typeof symptoms === 'string') return symptoms; // already free text
    if (!Array.isArray(symptoms) || symptoms.length === 0) return "";
    const symptomTexts = {
      A: "畏光/怕光", B: "眼部分泌物异常", C: "眼部有异物感或疼痛", D: "眼部肿胀",
      E: "头痛/恶心呕吐", F: "复视", G: "视野缺损", H: "视物变形",
      I: "眼球运动障碍", J: "看灯光有彩虹样光环", K: "眼前红色烟雾样遮挡",
      L: "色觉异常", M: "无其他明显症状"
    };
    return symptoms.map(s => symptomTexts[s] || s).join(", ");
  };

  // Translate gender enum to Chinese for display
  const toZhGender = (g) => {
    if (!g) return '';
    const s = String(g).trim().toLowerCase();
    if (s === 'male') return '男';
    if (s === 'female') return '女';
    if (s === 'other') return '其他';
    return g; // already human-entered or unknown, keep as-is
  };

  // Helpers
  const ensureEyeField = (data, eye) => {
    const eyeField = eye === 'left' ? 'leftEye' : eye === 'right' ? 'rightEye' : 'bothEyes';
    if (!data[eyeField]) data[eyeField] = {};
    return eyeField;
  };

  const getEyeData = (eye) => {
    if (!data) return null;
    if (eye === 'left') return data.leftEye || null;
    if (eye === 'right') return data.rightEye || null;
    if (eye === 'both') return data.bothEyes || null;
    return null;
  };

  const handleChange = (field, value) => {
    if (!onChange) return;
    onChange({ ...data, [field]: value });
  };

  const handleEyeChange = (eye, field, value) => {
    if (!onChange) return;
    const eyeField = eye === 'left' ? 'leftEye' : eye === 'right' ? 'rightEye' : 'bothEyes';
    const updatedEyeData = { ...(data[eyeField] || {}), [field]: value };
    onChange({ ...data, [eyeField]: updatedEyeData });
  };

  // NEW: intelligent toggle that preserves and pre-fills data
  const toggleAffectedArea = (area) => {
    const currentAreas = new Set(data.affectedArea || []);
    const nextData = { ...data, affectedArea: Array.from(currentAreas) };

    // Ensure fields exist but do NOT clear any existing data
    const leftField = 'leftEye';
    const rightField = 'rightEye';
    const bothField = 'bothEyes';

    // Initialize containers if missing (no wipe)
    if (!nextData[leftField]) nextData[leftField] = {};
    if (!nextData[rightField]) nextData[rightField] = {};
    if (!nextData[bothField]) nextData[bothField] = {};

    if (area === 'both') {
      if (currentAreas.has('both')) {
        // Deselect 'both' -> just remove the flag, keep bothEyes data
        nextData.affectedArea = [...currentAreas].filter(a => a !== 'both');
      } else {
        // Select 'both' -> remove individual flags
        nextData.affectedArea = ['both'];
        // Prefill bothEyes from the most recently non-empty eye
        const source = Object.keys(nextData[rightField] || {}).length ? nextData[rightField]
                      : Object.keys(nextData[leftField] || {}).length ? nextData[leftField]
                      : nextData[bothField];
        nextData[bothField] = { ...(source || {}) };
      }
    } else {
      // Individual eyes
      if (currentAreas.has(area)) {
        // Deselect one eye
        nextData.affectedArea = [...currentAreas].filter(a => a !== area);
      } else {
        // Selecting one eye removes 'both'
        const others = [...currentAreas].filter(a => a !== 'both');
        nextData.affectedArea = Array.from(new Set([...others, area]));
        const destField = area === 'left' ? leftField : rightField;
        // Prefill from bothEyes if dest empty
        if (Object.keys(nextData[destField] || {}).length === 0 && Object.keys(nextData[bothField] || {}).length > 0) {
          nextData[destField] = { ...nextData[bothField] };
        }
      }
    }

    onChange(nextData);
  };

  // Allow editing even when no data exists - create a fillable blank form
  const data = consultationData || {
    name: '',
    age: '',
    gender: '',
    phone: '',
    affectedArea: [],
    leftEye: {},
    rightEye: {},
    bothEyes: {},
    submissionTime: null
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-4 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">问诊信息 (Consultation Info)</h3>
        <button
          onClick={() => onSubmit && onSubmit()}
          disabled={isSubmitting}
          className={`px-3 py-1 rounded text-sm ${
            isSubmitting 
              ? "bg-blue-300 text-white cursor-not-allowed" 
              : "bg-blue-600 text-white hover:bg-blue-700"
          }`}
        >
          {isSubmitting ? "保存中..." : "保存 (Save)"}
        </button>
      </div>
      
      {!consultationData && (
        <div className="mb-3 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs text-yellow-800">
          <strong>提示：</strong>未找到已有问诊记录，可在此填写新的问诊信息
        </div>
      )}
      
      <div className="space-y-4 text-sm">
        {/* Basic Info */}
        <div className="space-y-2">
          <div className="flex">
            <span className="w-24 text-gray-600">姓名 (Name):</span>
            <input 
              type="text"
              value={data.name || ''}
              onChange={e => handleChange('name', e.target.value)}
              placeholder="请输入姓名"
              className="flex-1 border-b border-gray-300 focus:outline-none focus:border-blue-500 px-1"
            />
          </div>

          <div className="flex">
            <span className="w-24 text-gray-600">年龄 (Age):</span>
            <input
              type="text"
              value={data.age || ''}
              onChange={e => handleChange('age', e.target.value)}
              placeholder="例如 62"
              className="flex-1 border-b border-gray-300 focus:outline-none focus:border-blue-500 px-1"
            />
          </div>

          <div className="flex">
            <span className="w-24 text-gray-600">性别 (Gender):</span>
            <input
              type="text"
              value={toZhGender(data.gender || '')}
              onChange={e => handleChange('gender', e.target.value)}
              placeholder="男 / 女 / 其他"
              className="flex-1 border-b border-gray-300 focus:outline-none focus:border-blue-500 px-1"
            />
          </div>

          <div className="flex">
            <span className="w-24 text-gray-600">电话 (Phone):</span>
            <input 
              type="text"
              value={data.phone || ''}
              onChange={e => handleChange('phone', e.target.value)}
              placeholder="请输入联系电话"
              className="flex-1 border-b border-gray-300 focus:outline-none focus:border-blue-500 px-1"
            />
          </div>
        </div>
        
        {/* Affected Areas */}
        <div>
          <span className="block text-gray-600 font-medium mb-1">受累部位 (Affected Areas):</span>
          <div className="flex flex-wrap gap-3 pl-2">
            {['left', 'right', 'both'].map((area) => {
              const isSelected = (data.affectedArea || []).includes(area);
              const areaLabel = area === 'left' ? '左眼 (Left Eye)' : 
                              area === 'right' ? '右眼 (Right Eye)' : 
                              '双眼 (Both Eyes)';
              
              return (
                <label key={area} className="flex items-center cursor-pointer group">
                  <input 
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleAffectedArea(area)}
                    className="form-checkbox h-4 w-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                  />
                  <span className={`ml-2 text-sm transition-colors ${isSelected ? 'text-blue-700 font-medium' : 'text-gray-600 group-hover:text-blue-600'}`}>
                    {areaLabel}
                  </span>
                </label>
              );
            })}
          </div>
          <div className="mt-2 text-xs text-gray-500 italic">
            切换左右/双眼不会清空已填写内容；从“双眼”切换到单眼时，会预填入已填写的信息。
          </div>
        </div>
        
        {/* Eye-specific information */}
        {['left', 'right', 'both'].map(eye => {
          // Only show this eye section if it's in the affected areas
          if (!(data.affectedArea || []).includes(eye)) return null;
          
          const eyeData = getEyeData(eye) || {};
          const eyeLabel = eye === 'left' ? '左眼 (Left Eye)' : 
                          eye === 'right' ? '右眼 (Right Eye)' : 
                          '双眼 (Both Eyes)';
          
          return (
            <div key={eye} className="border-t border-gray-200 pt-3 mt-3">
              <div className="flex justify-between items-center">
                <h4 className="font-medium text-blue-600 mb-2">{eyeLabel}</h4>
              </div>
              
              <div className="space-y-2 pl-2">
                {/* Main Symptom */}
                <div>
                  <span className="block text-gray-600">主要症状 (Main Symptom):</span>
                  <textarea
                    value={
                      eyeData.mainSymptom
                        ? (eyeData.mainSymptom === 'I' && eyeData.mainSymptomOther
                            ? eyeData.mainSymptomOther
                            : (symptomMapping[eyeData.mainSymptom] || eyeData.mainSymptom))
                        : ''
                    }
                    onChange={(e) => handleEyeChange(eye, 'mainSymptom', e.target.value)}
                    className="w-full border rounded-md p-1 text-sm"
                    rows={2}
                  />
                </div>
                
                {/* Onset Method & Time */}
                <div>
                  <span className="block text-gray-600">症状起病方式及时间 (Onset):</span>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={eyeData.onsetMethod ? (onsetMapping[eyeData.onsetMethod] || eyeData.onsetMethod) : ''}
                      onChange={(e) => handleEyeChange(eye, 'onsetMethod', e.target.value)}
                      placeholder="起病方式"
                      className="flex-1 border rounded-md p-1 text-sm"
                    />
                    <input
                      type="text"
                      value={eyeData.onsetTime || ''}
                      onChange={(e) => handleEyeChange(eye, 'onsetTime', e.target.value)}
                      placeholder="起病时间"
                      className="flex-1 border rounded-md p-1 text-sm"
                    />
                  </div>
                </div>
                
                {/* Accompanying Symptoms */}
                <div>
                  <span className="block text-gray-600">伴随症状 (Accompanying):</span>
                  <textarea
                    value={formatAccompanyingSymptoms(eyeData.accompanyingSymptoms)}
                    onChange={(e) => handleEyeChange(eye, 'accompanyingSymptoms', e.target.value)}
                    className="w-full border rounded-md p-1 text-sm"
                    rows={2}
                  />
                </div>
                
                {/* Medical History */}
                <div>
                  <span className="block text-gray-600">既往病史及诱因 (History):</span>
                  <textarea
                    value={eyeData.medicalHistory ? (historyMapping[eyeData.medicalHistory] || eyeData.medicalHistory) : ''}
                    onChange={(e) => handleEyeChange(eye, 'medicalHistory', e.target.value)}
                    className="w-full border rounded-md p-1 text-sm"
                    rows={2}
                  />
                </div>
              </div>
            </div>
          );
        })}
        
        {/* Submission Time */}
        <div className="border-t border-gray-200 pt-2 text-xs text-gray-500">
          提交时间 (Submission Time): {data.submissionTime ? new Date(data.submissionTime).toLocaleString('zh-CN') : '新建记录'}
        </div>
      </div>
    </div>
  );
};

function App() {
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDisplayImages, setSelectedDisplayImages] = useState([]); // Stores image IDs for the 4 display slots
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');
  const [isExpandedImageModalOpen, setIsExpandedImageModalOpen] = useState(false); // New state for expanded image modal
  const [expandedImageInfo, setExpandedImageInfo] = useState(null); // Stores info of the image to expand
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false); // Track unsaved changes
  
  // New states for decoupled manual diagnosis
  const [manualDiagnosis, setManualDiagnosis] = useState({
    left_eye: {},
    right_eye: {}
  });
  const [customDiseases, setCustomDiseases] = useState({
    left_eye: '',
    right_eye: ''
  });
  const [diagnosisNotes, setDiagnosisNotes] = useState('');

  // Right-side LLM Chat (Demo)
  const [sideChatMessages, setSideChatMessages] = useState([]);
  const [sideChatInput, setSideChatInput] = useState('');
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmAbortCtrl, setLlmAbortCtrl] = useState(null);

  // Separate scroll area for chat
  const sideChatScrollRef = useRef(null);
  const [autoScrollChat, setAutoScrollChat] = useState(true);

  // ADD: 左侧AI容器ref + 右侧卡片高度
  const leftAIContainerRef = useRef(null);
  const [chatCardHeightPx, setChatCardHeightPx] = useState(null);

  // MOVE HERE: Declare autoStartRef BEFORE any code that uses it
  const autoStartRef = useRef({});

  // Add state for consultation info
  const [consultationData, setConsultationData] = useState(null);
  const [consultationDataEdited, setConsultationDataEdited] = useState(null);
  const [isConsultationSubmitting, setIsConsultationSubmitting] = useState(false);
  const [consultationSubmitMessage, setConsultationSubmitMessage] = useState('');
  
  // 添加缺失的状态变量
  const [currentPatientId, setCurrentPatientId] = useState('');
  const [patientNameSearch, setPatientNameSearch] = useState('');
  const [patientNameFromUrl, setPatientNameFromUrl] = useState(''); // Track patient_name from URL
  const [availablePatientNames, setAvailablePatientNames] = useState([]);
  const [showNameSuggestions, setShowNameSuggestions] = useState(false);
  const [sameNameConsultations, setSameNameConsultations] = useState([]);
  const [showConsultationSelector, setShowConsultationSelector] = useState(false);
  const [consultationLoading, setConsultationLoading] = useState(false);

  // New state for threshold management
  const [activeThresholdSet, setActiveThresholdSet] = useState(0); // 0 for set 1, 1 for set 2
  const [isAlteringThreshold, setIsAlteringThreshold] = useState(false);

  // New state for exam instance management
  const [availableExamInstances, setAvailableExamInstances] = useState([]); // List of available exam dates
  const [currentExamDate, setCurrentExamDate] = useState(null); // Currently selected exam date
  const [isLoadingInstances, setIsLoadingInstances] = useState(false);

  // Build backend URL and patient id
  const backendHost = window.location.hostname;
  const backendUrl = `http://${backendHost}:8000`;
  const urlParams = new URLSearchParams(window.location.search);
  const currentExamId = urlParams.get('ris_exam_id');

  // Prefer patientData.patient_id; fallback to patientData.id; else ris_exam_id
  const getCurrentPatientId = useMemo(() => {
    return () => (patientData?.patient_id || patientData?.id || currentExamId || '').toString();
  }, [patientData?.patient_id, patientData?.id, currentExamId]);

  const [llmConfig, setLlmConfig] = useState({ update_prompt: '' });


    // 添加：获取所有可用患者姓名
  const fetchAvailablePatientNames = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/api/consultation/names`, { cache: 'no-store' });
      const data = await res.json();
      setAvailablePatientNames(Array.isArray(data.patient_names) ? data.patient_names : []);
    } catch (e) {
      console.warn('Failed to fetch patient names:', e);
    }
  }, [backendUrl]);

  // 添加：根据 ris_exam_id（可选 patient_name）获取问诊信息，并初始化可编辑副本
  const fetchConsultationInfo = useCallback(async (risExamId, searchName = null, autoPickLatestIfMultiple = false) => {
    if (!risExamId) return;
    setConsultationLoading(true);
    try {
      const url = searchName
        ? `${backendUrl}/api/consultation/${encodeURIComponent(risExamId)}?patient_name=${encodeURIComponent(searchName)}`
        : `${backendUrl}/api/consultation/${encodeURIComponent(risExamId)}`;
      const res = await fetch(url, { cache: 'no-store' });
      const data = await res.json();

      if (data.status === 'multiple_matches') {
        const candidates = Array.isArray(data.same_name_consultations) ? [...data.same_name_consultations] : [];

        if (autoPickLatestIfMultiple && candidates.length > 0) {
          // 选最新：按提交时间倒序
          candidates.sort((a, b) => {
            const ta = a?.submissionTime ? new Date(a.submissionTime).getTime() : 0;
            const tb = b?.submissionTime ? new Date(b.submissionTime).getTime() : 0;
            return tb - ta;
          });
          const first = candidates[0];

          // 直接获取该索引的问诊详情（避免依赖组件内其他函数，减少TDZ风险）
          try {
            const byIdxUrl = `${backendUrl}/api/consultation/${encodeURIComponent(risExamId)}/by_index/${first.index}?use_refined=${!!first.isRefined}`;
            const r = await fetch(byIdxUrl, { cache: 'no-store' });
            const picked = await r.json();
            if (picked.status === 'success_refined' || picked.status === 'success_original') {
              setConsultationData(picked.consultation_data || null);
              setConsultationDataEdited(picked.consultation_data ? JSON.parse(JSON.stringify(picked.consultation_data)) : null);
              setShowConsultationSelector(false);
              setSameNameConsultations([]);
            } else {
              // 回退到选择器
              setSameNameConsultations(candidates);
              setShowConsultationSelector(true);
              setConsultationData(null);
              setConsultationDataEdited(null);
            }
          } catch {
            // 回退到选择器
            setSameNameConsultations(candidates);
            setShowConsultationSelector(true);
            setConsultationData(null);
            setConsultationDataEdited(null);
          }
        } else {
          // 保持原行为：展示同名选择器
          setSameNameConsultations(candidates);
          setShowConsultationSelector(true);
          setConsultationData(null);
          setConsultationDataEdited(null);
        }
      } else if (data.status === 'success_refined' || data.status === 'success_original') {
        setConsultationData(data.consultation_data || null);
        setConsultationDataEdited(data.consultation_data ? JSON.parse(JSON.stringify(data.consultation_data)) : null);
        setShowConsultationSelector(false);
        setSameNameConsultations([]);
      } else {
        setConsultationData(null);
        setConsultationDataEdited({
          name: '',
          age: '',
          gender: '',
          phone: '',
          affectedArea: [],
          leftEye: {},
          rightEye: {},
          bothEyes: {},
          submissionTime: null
        }); // Initialize with blank form when no consultation exists
        setShowConsultationSelector(false);
        setSameNameConsultations([]);
      }
    } catch (e) {
      console.error('Failed to fetch consultation info:', e);
      setConsultationData(null);
      setConsultationDataEdited(null);
    } finally {
      setConsultationLoading(false);
    }
  }, [backendUrl]);

  // 添加：姓名输入/选择逻辑
  const handlePatientNameSearch = useCallback((val) => {
    setPatientNameSearch(val);
    setShowNameSuggestions(!!val && availablePatientNames.some(n => n.includes(val)));
  }, [availablePatientNames]);

  const selectPatientName = useCallback((name) => {
    setPatientNameSearch(name);
    setShowNameSuggestions(false);
    if (currentPatientId) {
      fetchConsultationInfo(currentPatientId, name);
    }
  }, [currentPatientId, fetchConsultationInfo]);

  // Fetch available exam instances for a patient - MOVED HERE to avoid TDZ error
  const fetchExamInstances = useCallback(async (examId) => {
    if (!examId) return;
    
    setIsLoadingInstances(true);
    try {
      const response = await fetch(`${backendUrl}/api/patients/${examId}/instances`);
      if (!response.ok) {
        throw new Error(`Failed to fetch exam instances: ${response.status}`);
      }
      const result = await response.json();
      setAvailableExamInstances(result.instances || []);
      // If we have instances and no current exam date is set, set it to the first (latest)
      if (result.instances && result.instances.length > 0 && !currentExamDate) {
        setCurrentExamDate(result.instances[0].exam_date);
      }
    } catch (err) {
      console.error('Error fetching exam instances:', err);
      setAvailableExamInstances([]);
    } finally {
      setIsLoadingInstances(false);
    }
  }, [backendUrl, currentExamDate]);

  // Fetch patient data - MOVED HERE to avoid TDZ error
  const fetchPatientData = useCallback(async (examId, examDate = null) => {
    if (!examId) {
      setError('No ris_exam_id provided in URL. Please add ?ris_exam_id=<exam_id> to the URL.');
      setLoading(false);
      return;
    }

    console.log(`Starting to fetch patient data for exam ID: ${examId}${examDate ? ` (date: ${examDate})` : ''}`);
    const startTime = performance.now();
    setLoading(true);
    setError(null);

    try {
      // Build URL with optional exam_date parameter
      const url = examDate 
        ? `${backendUrl}/api/patients/${examId}?exam_date=${examDate}`
        : `${backendUrl}/api/patients/${examId}`;
      
      console.log(`Fetching from: ${url}`);
      const response = await fetch(url);
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error(`Examination with ID ${examId} not found.`);
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      const fetchTime = performance.now() - startTime;
      console.log(`Successfully fetched patient data in ${fetchTime.toFixed(2)}ms`);

      // Store a pristine copy of the data for tracking changes.
      const dataWithOriginal = { ...result, original: JSON.parse(JSON.stringify(result)) };

      setPatientData(dataWithOriginal);
      
      // Set the active threshold set from patient data
      setActiveThresholdSet(result.active_threshold_set || 0);

      // Initialize the 4 display images in a fixed desired order with graceful fallback
      const desiredOrder = ['左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'];
      const allImages = Array.isArray(result.eye_images) ? result.eye_images : [];
      const selectedIds = [];
      const used = new Set();

      // First pass: pick images matching desired types in order
      for (const desiredType of desiredOrder) {
        const match = allImages.find(img => img.type === desiredType && !used.has(img.id));
        if (match) {
          selectedIds.push(match.id);
          used.add(match.id);
        } else {
          selectedIds.push(null);
        }
      }

      // Fill null slots with any remaining images
      for (let i = 0; i < selectedIds.length; i++) {
        if (!selectedIds[i]) {
          const fallback = allImages.find(img => !used.has(img.id));
          if (fallback) {
            selectedIds[i] = fallback.id;
            used.add(fallback.id);
          }
        }
      }

      setSelectedDisplayImages(selectedIds.filter(id => id !== null));
      setLoading(false);
      setHasUnsavedChanges(false);

      // Load manual diagnosis data if available, otherwise initialize with AI predictions
      try {
        const manualResp = await fetch(`${backendUrl}/api/manual_diagnosis/${examId}`);
        let manualDataLoaded = false;
        
        if (manualResp.ok) {
          const manualData = await manualResp.json();
          
          // Check if manual_diagnosis exists AND has actual data (not just empty objects)
          const hasManualDiagnosisData = manualData.manual_diagnosis && (
            (manualData.manual_diagnosis.left_eye && Object.keys(manualData.manual_diagnosis.left_eye).length > 0) ||
            (manualData.manual_diagnosis.right_eye && Object.keys(manualData.manual_diagnosis.right_eye).length > 0)
          );
          
          if (hasManualDiagnosisData) {
            // Ensure the structure has left_eye and right_eye objects
            const validatedDiagnosis = {
              left_eye: manualData.manual_diagnosis.left_eye || {},
              right_eye: manualData.manual_diagnosis.right_eye || {}
            };
            setManualDiagnosis(validatedDiagnosis);
            manualDataLoaded = true;
          }
          if (manualData.custom_diseases) {
            setCustomDiseases(manualData.custom_diseases);
          }
          if (manualData.diagnosis_notes) {
            setDiagnosisNotes(manualData.diagnosis_notes);
          }
        }
        
        // If no saved manual diagnosis exists, initialize with AI predictions
        if (!manualDataLoaded) {
          // List of disease keys that should be checked in manual diagnosis
          const diseaseKeys = ['青光眼', '糖网', 'AMD', '病理性近视', '高度近视', 'RVO', 'RAO', '视网膜脱离', '其它视网膜病', '其它黄斑病变', '白内障', '正常'];
          
          // Helper to calculate score (probability relative to threshold)
          const calculateScore = (prob, threshold) => {
            if (prob >= threshold * 2) return prob / (threshold * 2);
            if (prob >= threshold) return 0.5 + (prob - threshold) / threshold * 0.3;
            if (prob >= threshold * 0.5) return 0.2 + (prob - threshold * 0.5) / (threshold * 0.5) * 0.3;
            return prob / (threshold * 0.5) * 0.2;
          };
          
          const getPrimaryAIDisease = (data, eyeKey) => {
            const preds = data?.prediction_results?.[eyeKey];
            if (!preds) return null;

            const thresholds = (data?.prediction_thresholds && data.prediction_thresholds) || {};
            let best = null;

            Object.entries(preds).forEach(([diseaseKey, probValue]) => {
              // Only consider diseases in our manual disease list
              if (!diseaseKeys.includes(diseaseKey)) return;
              
              const threshold = typeof thresholds[diseaseKey] === 'number' ? thresholds[diseaseKey] : 0.5;
              const prob = typeof probValue === 'number' ? probValue : 0;
              const score = calculateScore(prob, threshold);

              if (!best || score > best.score) {
                best = { diseaseKey, score };
              }
            });

            return best?.diseaseKey || null;
          };

          const buildInitialManualDiagnosis = (eyeKey) => {
            const primaryDisease = getPrimaryAIDisease(result, eyeKey);
            return diseaseKeys.reduce((acc, diseaseKey) => {
              acc[diseaseKey] = primaryDisease === diseaseKey;
              return acc;
            }, {});
          };

          const initialManualDiagnosis = {
            left_eye: buildInitialManualDiagnosis('left_eye'),
            right_eye: buildInitialManualDiagnosis('right_eye')
          };
          
          console.log('Initializing manual diagnosis with AI predictions:', initialManualDiagnosis);
          setManualDiagnosis(initialManualDiagnosis);
          setCustomDiseases({ left_eye: '', right_eye: '' });
          setDiagnosisNotes('');
        }
      } catch (e) {
        console.warn('Failed to load manual diagnosis data:', e);
        // Even if loading fails, try to initialize with AI predictions
        const diseaseKeys = ['青光眼', '糖网', 'AMD', '病理性近视', '高度近视', 'RVO', 'RAO', '视网膜脱离', '其它视网膜病', '其它黄斑病变', '白内障', '正常'];
        
        const calculateScore = (prob, threshold) => {
          if (prob >= threshold * 2) return prob / (threshold * 2);
          if (prob >= threshold) return 0.5 + (prob - threshold) / threshold * 0.3;
          if (prob >= threshold * 0.5) return 0.2 + (prob - threshold * 0.5) / (threshold * 0.5) * 0.3;
          return prob / (threshold * 0.5) * 0.2;
        };
        
        const getPrimaryAIDisease = (data, eyeKey) => {
          const preds = data?.prediction_results?.[eyeKey];
          if (!preds) return null;

          const thresholds = (data?.prediction_thresholds && data.prediction_thresholds) || {};
          let best = null;

          Object.entries(preds).forEach(([diseaseKey, probValue]) => {
            if (!diseaseKeys.includes(diseaseKey)) return;
            const threshold = typeof thresholds[diseaseKey] === 'number' ? thresholds[diseaseKey] : 0.5;
            const prob = typeof probValue === 'number' ? probValue : 0;
            const score = calculateScore(prob, threshold);

            if (!best || score > best.score) {
              best = { diseaseKey, score };
            }
          });

          return best?.diseaseKey || null;
        };

        const buildInitialManualDiagnosis = (eyeKey) => {
          const primaryDisease = getPrimaryAIDisease(result, eyeKey);
          return diseaseKeys.reduce((acc, diseaseKey) => {
            acc[diseaseKey] = primaryDisease === diseaseKey;
            return acc;
          }, {});
        };

        const initialManualDiagnosis = {
          left_eye: buildInitialManualDiagnosis('left_eye'),
          right_eye: buildInitialManualDiagnosis('right_eye')
        };
        
        console.log('Initializing manual diagnosis with AI predictions (from catch):', initialManualDiagnosis);
        setManualDiagnosis(initialManualDiagnosis);
        setCustomDiseases({ left_eye: '', right_eye: '' });
        setDiagnosisNotes('');
      }

    } catch (error) {
      console.error('Error fetching patient data:', error);
      setError(error.message);
      setLoading(false);
    }
  }, [backendUrl]);

  // 首次加载：取可用姓名列表
  useEffect(() => {
    fetchAvailablePatientNames();
  }, [fetchAvailablePatientNames]);

  // 已有：根据 URL 初始化 ris_exam_id 与按姓名检索
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const risExamIdFromUrl = urlParams.get('ris_exam_id');
    const patientNameFromUrl = urlParams.get('patient_name');

    if (risExamIdFromUrl) {
      setCurrentPatientId(risExamIdFromUrl);
      if (patientNameFromUrl) {
        const decoded = decodeURIComponent(patientNameFromUrl);
        setPatientNameSearch(decoded);
        setPatientNameFromUrl(decoded); // Store patient_name from URL
        fetchConsultationInfo(risExamIdFromUrl, decoded, true); // 同名多条时自动选最新
      } else {
        setPatientNameFromUrl(''); // No patient_name in URL
        fetchConsultationInfo(risExamIdFromUrl);
      }
      // Fetch exam instances first, then load patient data
      fetchExamInstances(risExamIdFromUrl);
      fetchPatientData(risExamIdFromUrl);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 修正：根据索引选择特定问诊记录（使用 backendUrl，并同步可编辑副本）
  const selectConsultationByIndex = async (index, useRefined = true) => {
    try {
      const response = await fetch(`${backendUrl}/api/consultation/${currentPatientId}/by_index/${index}?use_refined=${useRefined}`);
      const data = await response.json();

      if (data.status === 'success_refined' || data.status === 'success_original') {
        setConsultationData(data.consultation_data);
        setConsultationDataEdited(data.consultation_data ? JSON.parse(JSON.stringify(data.consultation_data)) : null);
        setShowConsultationSelector(false);
        setSameNameConsultations([]);
      }
    } catch (error) {
      console.error('Failed to fetch consultation by index:', error);
    }
  };

  // 修正：保存问诊时同时传 patient_id 与 ris_exam_id 以兼容后端两种模型
  const handleConsultationSubmit = useCallback(async () => {
    if (!consultationDataEdited || !currentExamId) return;

    setIsConsultationSubmitting(true);
    setConsultationSubmitMessage('');

    try {
      // Prepare consultation data with proper name handling
      const submissionData = { ...consultationDataEdited };
      
      // If creating new consultation and name is empty, use patient_name from URL or "未知患者"
      if (!consultationData && !submissionData.name) {
        submissionData.name = patientNameFromUrl || patientData?.name || '未知患者';
      }
      
      // Add timestamp for new consultations
      if (!consultationData) {
        submissionData.submissionTime = new Date().toISOString();
      }

      const response = await fetch(`${backendUrl}/api/consultation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: currentExamId,     // 兼容旧后端
          ris_exam_id: currentExamId,    // 兼容新后端
          consultation_data: submissionData
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const result = await response.json();
      setConsultationSubmitMessage(result.status || 'Consultation data saved!');
      setConsultationData(submissionData);
      setConsultationDataEdited(JSON.parse(JSON.stringify(submissionData)));
    } catch (e) {
      console.error("Failed to submit consultation data:", e);
      setConsultationSubmitMessage(`Error: ${e.message}`);
    } finally {
      setIsConsultationSubmitting(false);
      setTimeout(() => setConsultationSubmitMessage(''), 3000);
    }
  }, [backendUrl, currentExamId, consultationDataEdited, consultationData, patientNameFromUrl, patientData]);

  // Handle alter threshold
  const handleAlterThreshold = useCallback(async () => {
    if (!currentExamId) return;

    setIsAlteringThreshold(true);
    try {
      const response = await fetch(`${backendUrl}/api/alter_threshold`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          patient_id: currentExamId,
          exam_date: currentExamDate || null
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const result = await response.json();
      
      // Update the active threshold set
      setActiveThresholdSet(result.active_threshold_set);
      
      // Update patient data with new thresholds and diagnosis results
      setPatientData(prevData => ({
        ...prevData,
        prediction_thresholds: result.new_thresholds,
        diagnosis_results: result.updated_diagnosis_results,
        active_threshold_set: result.active_threshold_set
      }));

      // Show success message temporarily
      setSubmitMessage(`阈值已更新至套装 ${result.active_threshold_set + 1} (Thresholds updated to set ${result.active_threshold_set + 1})`);
      setTimeout(() => setSubmitMessage(''), 3000);

    } catch (error) {
      console.error('Failed to alter threshold:', error);
      setSubmitMessage(`阈值更新失败: ${error.message}`);
      setTimeout(() => setSubmitMessage(''), 3000);
    } finally {
      setIsAlteringThreshold(false);
    }
  }, [backendUrl, currentExamId]);

  // Handle switching exam dates
  const handleExamDateSwitch = useCallback(async (examDate) => {
    if (!currentExamId || examDate === currentExamDate) return;
    
    setCurrentExamDate(examDate);
    // Reload patient data for the selected exam date
    await fetchPatientData(currentExamId, examDate);
  }, [currentExamId, currentExamDate, fetchPatientData]);

  // Load LLM prompts config (update_prompt)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${backendUrl}/api/llm_config`, { cache: 'no-store' });
        if (res.ok) setLlmConfig(await res.json());
      } catch {}
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendUrl]);

  // Auto-trigger LLM regeneration once per patient when chat is empty
  useEffect(() => {
    const pid = getCurrentPatientId();
    if (!pid) return;

    // 已触发过则不再重复
    if (autoStartRef.current[pid]) return;

    // 条件：有患者数据，未在生成中，对话为空
    if (patientData && !llmLoading && sideChatMessages.length === 0) {
      autoStartRef.current[pid] = true;
      regenerateSideOpinion(); // 开始流式生成
    }
    // 仅监听必要的状态，避免依赖未初始化的 const
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientData, llmLoading, sideChatMessages.length, getCurrentPatientId]);
  

  // Chat scroll handling (only autoscroll if user is at bottom)
  useEffect(() => {
    const el = sideChatScrollRef.current;
    if (!el || !autoScrollChat) return;
    el.scrollTop = el.scrollHeight;
  }, [sideChatMessages, autoScrollChat]);

  const handleSideChatScroll = () => {
    const el = sideChatScrollRef.current;
    if (!el) return;
    const threshold = 32;
    const atBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) <= threshold;
    setAutoScrollChat(atBottom);
  };

  // ADD: 同步右侧卡片高度 = 左侧 AI（摘要+概率）总高度
  useLayoutEffect(() => {
    const el = leftAIContainerRef.current;
    if (!el) return;

    const sync = () => {
      const h = Math.max(0, Math.round(el.getBoundingClientRect().height));
      setChatCardHeightPx(h || null);
    };

    // 初次同步
    sync();

    // 监听左侧高度变化与窗口尺寸变化
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    window.addEventListener('resize', sync);

    return () => {
      try { ro.disconnect(); } catch {}
      window.removeEventListener('resize', sync);
    };
  }, []);

  // Streaming send helper for LLM side chat
  const sendSideChatStreaming = async (text, opts = {}) => {
    const q = (text ?? sideChatInput).trim();
    const reset = !!opts.reset;
    if (!q || llmLoading) return;

    const existingMessages = reset ? [] : sideChatMessages;

    if (text === undefined) setSideChatInput('');

    const history = [
      ...existingMessages
        .filter(m => m && typeof m.role === 'string' && typeof m.content === 'string' && ['user', 'assistant'].includes(m.role))
        .map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: q }
    ];

    // Append placeholder for streaming assistant reply
    setSideChatMessages(() => [
      ...history,
      { role: 'assistant', content: '' },
    ]);

    const controller = new AbortController();
    setLlmAbortCtrl(controller);
    setLlmLoading(true);

    let gotAny = false;

    try {
      const payload = {
        patient_id: getCurrentPatientId(),
        patient_name: patientNameSearch || null,  // Include patient_name for consultation matching
        messages: history,
      };

      const resp = await fetch(`${backendUrl}/api/llm_chat_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'text/plain' },
        body: JSON.stringify(payload),
        signal: controller.signal,
        cache: 'no-store',
      });

      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (!chunk || chunk.trim().length === 0) continue;
        gotAny = true;
        setSideChatMessages(prev => {
          const msgs = [...prev];
          let idx = msgs.length - 1;
          while (idx >= 0 && msgs[idx].role !== 'assistant') idx--;
          if (idx >= 0) msgs[idx] = { ...msgs[idx], content: (msgs[idx].content || '') + chunk };
          return msgs;
        });
      }
    } catch (e) {
      if (e.name !== 'AbortError' && !gotAny) {
        setSideChatMessages(prev => {
          const msgs = [...prev];
          let idx = msgs.length - 1;
          while (idx >= 0 && msgs[idx].role !== 'assistant') idx--;
          if (idx >= 0) msgs[idx] = { ...msgs[idx], content: '（LLM服务暂不可用，已结束演示流式输出。）' };
          else msgs.push({ role: 'assistant', content: '（LLM服务暂不可用，已结束演示流式输出。）' });
          return msgs;
        });
      }
    } finally {
      setLlmLoading(false);
      setLlmAbortCtrl(null);
    }
  };

  // Update button: clear UI, then regenerate
  const regenerateSideOpinion = async () => {
    const prompt =
      (llmConfig?.update_prompt && llmConfig.update_prompt.trim()) ||
      '请基于最新问诊信息、AI预测与人工复检结果，生成简要且可操作的临床意见摘要。';

    setSideChatMessages([]);     // ensure UI reset immediately
    setAutoScrollChat(true);
    await sendSideChatStreaming(prompt, { reset: true });
  };

  // Remove demo seeding: start with blank chat
  // useEffect(() => {
  //   if (patientData && sideChatMessages.length === 0) {
  //     setSideChatMessages([
  //       {
  //         role: 'assistant',
  //         content:
  //           '临床助手（演示）：【诊断推理过程】\n' +
  //           '患者为 62 岁女性，高血压和糖尿病史，主诉双眼渐进性视物模糊 4 个月伴视物变形，结合神经网络模型对 CFP 的预测结果：右眼明确为年龄相关性黄斑病变（AMD），与患者年龄、典型视物变形症状契合，且高血压、糖尿病可能诱发 AMD 进展及视网膜动脉阻塞，需排除其他黄斑病变干扰；左眼 “其他黄斑病变接近阈值”，虽症状与右眼一致但病变类型未明，需排查非 AMD 类黄斑病变，同时因全身血管疾病，需关注左眼早期 AMD 及视网膜动脉阻塞的隐匿风险。'
  //       },
  //       {
  //         role: 'assistant',
  //         content:
  //           '临床助手（演示）：【检查与治疗建议】\n' +
  //           '优先完善 OCT（明确双眼黄斑结构异常）、FFA（评估视网膜血管及右眼 AMD 类型），监测血压、血糖、血脂；治疗上，右眼干性 AMD 口服抗氧化剂，湿性 AMD 行抗 VEGF 注射，左眼暂不针对性用药，口服改善微循环药预防动脉阻塞（告知突发视力下降 1 小时内急诊），同时控制饮食、适度运动、外出戴防蓝光镜；随访需左眼每 1-2 个月查 OCT，双眼每 3 个月查 CFP，动态调整诊疗方案。'
  //       }
  //     ]);
  //   }
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, [patientData]); // seed once per patient


  // NEW: fetch LLM prompts/config once
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${backendUrl}/api/llm_config`, { cache: 'no-store' });
        if (res.ok) {
          const cfg = await res.json();
          setLlmConfig(prev => ({ ...prev, ...cfg }));
        }
      } catch (e) {
        console.warn('Failed to load LLM config; using defaults.', e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendUrl]);

  const patientIdPrefix = "病例索引 (Case Index): ";

  // Enhanced disease mapping with both Chinese and English names
  const diseaseInfo = {
    青光眼: { 
      chinese: '青光眼', 
      english: 'Glaucoma',
      fullName: '青光眼 (Glaucoma)',
      shortName: 'Glaucoma',
      category: 'glaucoma',
      color: 'text-purple-600'
    },
    糖网: { 
      chinese: '糖网', 
      english: 'Diabetic Retinopathy',
      fullName: '糖网 (Diabetic Retinopathy)',
      shortName: 'DR',
      category: 'retinal',
      color: 'text-red-600'
    },
    AMD: { 
      chinese: '年龄相关性黄斑变性', 
      english: 'Age-related Macular Degeneration',
      fullName: '年龄相关性黄斑变性 (Age-related Macular Degeneration)',
      shortName: 'AMD',
      category: 'macular',
      color: 'text-orange-600'
    },
    病理性近视: { 
      chinese: '病理性近视', 
      english: 'Pathological Myopia',
      fullName: '病理性近视 (Pathological Myopia)',
      shortName: 'PM',
      category: 'macular',
      color: 'text-yellow-600'
    },
    RVO: { 
      chinese: '视网膜静脉阻塞', 
      english: 'Retinal Vein Occlusion',
      fullName: '视网膜静脉阻塞 (Retinal Vein Occlusion)',
      shortName: 'RVO',
      category: 'vascular',
      color: 'text-pink-600'
    },
    RAO: { 
      chinese: '视网膜动脉阻塞', 
      english: 'Retinal Artery Occlusion',
      fullName: '视网膜动脉阻塞 (Retinal Artery Occlusion)',
      shortName: 'RAO',
      category: 'vascular',
      color: 'text-rose-600'
    },
    视网膜脱离: { 
      chinese: '视网膜脱离', 
      english: 'Retinal Detachment',
      fullName: '视网膜脱离 (Retinal Detachment)',
      shortName: 'RD',
      category: 'retinal',
      color: 'text-indigo-600'
    },
    其它视网膜病: { 
      chinese: '其它视网膜病', 
      english: 'Other Retinal Diseases',
      fullName: '其它视网膜病 (Other Retinal)',
      shortName: 'Other Retinal',
      category: 'retinal',
      color: 'text-cyan-600'
    },
    其它黄斑病变: { 
      chinese: '其它黄斑病变', 
      english: 'Other Macular Diseases',
      fullName: '其它黄斑病变 (Other Macular)',
      shortName: 'Other Macular',
      category: 'macular',
      color: 'text-teal-600'
    },
    其它眼底病变: {
      chinese: '其它眼底病变',
      english: 'Other Fundus Diseases',
      fullName: '其它眼底病变 (Other Fundus Diseases)',
      shortName: 'Other Fundus',
      category: 'fundus',
      color: 'text-teal-700'
    },
    白内障: { 
      chinese: '白内障', 
      english: 'Cataract',
      fullName: '白内障 (Cataract)',
      shortName: 'Cataract',
      category: 'lens',
      color: 'text-blue-600'
    },
    正常: { 
      chinese: '正常', 
      english: 'Normal',
      fullName: '正常 (Normal)',
      shortName: 'Normal',
      category: 'normal',
      color: 'text-green-600'
    }
  };

  // Manual diagnosis diseases (decoupled from AI predictions)
  const manualDiseaseInfo = {
    青光眼: { 
      chinese: '青光眼', 
      english: 'Glaucoma',
      fullName: '青光眼 (Glaucoma)',
      shortName: 'Glaucoma'
    },
    糖网: { 
      chinese: '糖网', 
      english: 'Diabetic Retinopathy',
      fullName: '糖网 (Diabetic Retinopathy)',
      shortName: 'DR'
    },
    AMD: { 
      chinese: '年龄相关性黄斑变性', 
      english: 'Age-related Macular Degeneration',
      fullName: '年龄相关性黄斑变性 (Age-related Macular Degeneration)',
      shortName: 'AMD'
    },
    病理性近视: { 
      chinese: '病理性近视', 
      english: 'Pathological Myopia',
      fullName: '病理性近视 (Pathological Myopia)',
      shortName: 'PM'
    },
    高度近视: { 
      chinese: '豹纹状眼底', 
      english: 'Leopard Fundus',
      fullName: '豹纹状眼底 (Leopard Fundus)',
      shortName: 'Leopard Fundus'
    },
    RVO: { 
      chinese: '视网膜静脉阻塞', 
      english: 'Retinal Vein Occlusion',
      fullName: '视网膜静脉阻塞 (Retinal Vein Occlusion)',
      shortName: 'RVO'
    },
    RAO: { 
      chinese: '视网膜动脉阻塞', 
      english: 'Retinal Artery Occlusion',
      fullName: '视网膜动脉阻塞 (Retinal Artery Occlusion)',
      shortName: 'RAO'
    },
    视网膜脱离: { 
      chinese: '视网膜脱离', 
      english: 'Retinal Detachment',
      fullName: '视网膜脱离 (Retinal Detachment)',
      shortName: 'RD'
    },
    白内障: { 
      chinese: '白内障', 
      english: 'Cataract',
      fullName: '白内障 (Cataract)',
      shortName: 'Cataract'
    },
    正常: { 
      chinese: '正常', 
      english: 'Normal',
      fullName: '正常 (Normal)',
      shortName: 'Normal'
    }
  };

  // Keep the old diseaseNames for backward compatibility
  const diseaseNames = Object.keys(diseaseInfo).reduce((acc, key) => {
    acc[key] = diseaseInfo[key].english;
    return acc;
  }, {});

  // Order of diseases for display (match backend model fields)
  const diseaseOrder = [
    '青光眼','糖网','AMD','病理性近视','RVO','RAO','视网膜脱离','其它视网膜病','其它黄斑病变','白内障','正常'
  ];

  const summaryMergeGroups = [
    { key: '其它眼底病变', members: ['其它视网膜病', '其它黄斑病变'] }
  ];

  // Map raw probability p (0-1) to visual width so that threshold t maps to 0.5
  // Linear piecewise: [0,t] -> [0,0.5], [t,1] -> [0.5,1]. This keeps monotonicity.
  const mapProbToWidth = (p, t) => {
    if (t <= 0) return p; // avoid divide by zero
    if (t >= 1) return 1; // degenerate
    if (p <= t) {
      return (p / (t * 2));
    }
    return 0.5 + (p - t) / (2 * (1 - t));
  };

  const formatProb = (p) => (p === undefined || p === null ? '--' : p.toFixed(2));

  const getSummaryRankingItems = (eyeKey) => {
    const preds = patientData?.prediction_results?.[eyeKey];
    const thresholds = patientData?.prediction_thresholds || {};
    if (!preds) return [];

    const rawItems = diseaseOrder.map((dk) => {
      const p = preds[dk] ?? 0;
      const t = thresholds[dk] ?? 0.5;
      return {
        key: dk,
        sourceKey: dk,
        p,
        t,
        score: mapProbToWidth(p, t),
      };
    });

    const rawMap = new Map(rawItems.map((item) => [item.key, item]));
    const skipped = new Set();
    const merged = [];

    summaryMergeGroups.forEach((group) => {
      const candidates = group.members
        .map((member) => rawMap.get(member))
        .filter(Boolean);
      if (candidates.length > 0) {
        const best = candidates.reduce((acc, curr) => (curr.score > acc.score ? curr : acc));
        merged.push({ ...best, key: group.key, sourceKey: best.key });
        candidates.forEach((item) => skipped.add(item.key));
      }
    });

    rawItems.forEach((item) => {
      if (!skipped.has(item.key)) {
        merged.push(item);
      }
    });

    merged.sort((a, b) => b.score - a.score);
    return merged;
  };

  // Helper function to deep compare two objects
  const isDataEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  // Build a natural Chinese summary per eye; skip '正常' in secondary mentions
  const buildEyeSummary = (eyeKey) => {
    try {
      const ranked = getSummaryRankingItems(eyeKey);
      if (ranked.length === 0) return '';
      const eyeLabel = eyeKey === 'left_eye' ? '左眼' : '右眼';
      const topThree = ranked.slice(0, 3);
      const [top] = topThree;
      if (!top) return '';

      const topName = diseaseInfo[top.key]?.chinese || top.key;

      // Choose a natural phrase based on relation to threshold
      const t = top.t ?? 0.5;
      const p = top.p ?? 0;
      let probPhrase = '风险偏高';
      if (p >= t * 1.2) probPhrase = '风险明显偏高';
      else if (p >= t) probPhrase = '风险较高';
      else if (p >= t * 0.8) probPhrase = '风险较高';
      else probPhrase = '可能存在风险';

      // If top is 正常, use a more natural normal-first sentence
      if (top.key === '正常') {
        const others = topThree
          .slice(1)
          .filter((x) => x.key !== '正常' && (x.p ?? 0) >= 0.2 * (x.t ?? 0.5) && x.score > 0)
          .map((x) => diseaseInfo[x.key]?.chinese || x.key);

        if (others.length === 0) {
          return `患者${eyeLabel}整体倾向于正常，目前未见明显异常信号。`;
        }
        if (others.length === 1) {
          return `患者${eyeLabel}整体倾向于正常，但建议关注${others[0]}的可能性。`;
        }
        return `患者${eyeLabel}整体倾向于正常，但建议关注${others[0]}与${others[1]}的可能性。`;
      }

      // Build secondary mentions, skipping '正常'
      const others = topThree
        .slice(1)
        .filter((x) => x.key !== '正常' && (x.p ?? 0) >= 0.2 * (x.t ?? 0.5) && x.score > 0)
        .map((x) => diseaseInfo[x.key]?.chinese || x.key);

      // Main sentence
      let sentence = `患者${eyeLabel}${topName}的${probPhrase}`;

      // Tail based on others
      if (others.length === 0) {
        sentence += '。其余疾病可能性总体较低，建议结合临床综合评估。';
      } else if (others.length === 1) {
        sentence += `，还应着重关注${others[0]}的可能性。`;
      } else {
        sentence += `，还应着重关注${others[0]}与${others[1]}的可能性。`;
      }

      return sentence;
    } catch (e) {
      console.warn('Failed to build eye summary:', e);
      return '';
    }
  };

  // Compute highlight groups per eye
  // - Primaries: all diseases crossing threshold (p >= t); if '正常' crosses, show only '正常'
  // - Secondaries: up to 2 diseases (exclude '正常' and primaries) crossing half-threshold (p >= 0.5*t)
  const getEyeHighlights = (eyeKey) => {
    const ranked = getSummaryRankingItems(eyeKey);
    if (ranked.length === 0) return { primaries: [], secondaries: [] };

    const topThree = ranked.slice(0, 3);

    const enrich = (item) => ({
      key: item.key,
      name: diseaseInfo[item.key]?.chinese || item.key,
      p: item.p,
      t: item.t,
      status: item.key === '正常'
        ? '正常'
        : (item.p >= item.t * 1.2 ? '明显偏高' : (item.p >= item.t ? '较高' : (item.p >= item.t * 0.8 ? '接近阈值' : '较低')))
    });

    const primaryItem = topThree[0];
    const primaries = primaryItem ? [enrich(primaryItem)] : [];
    const secondaries = topThree
      .slice(1)
      .filter((item) => item.key !== '正常' && item.score > 0)
      .map(enrich);

    return { primaries, secondaries };
  };

  // Handler for changes to Image Type or Image Quality dropdowns
  const handleImageDetailChange = (imageId, field, value) => {
    setPatientData(prevData => {
      if (!prevData) return null;
      const updatedEyeImages = prevData.eye_images.map(img =>
        img.id === imageId ? { ...img, [field]: value } : img
      );
      setHasUnsavedChanges(true);
      return { ...prevData, eye_images: updatedEyeImages };
    });
  };

  // Handler for manual diagnosis (decoupled from AI predictions)
  const handleManualDiagnosisToggle = (eye, disease) => {
    setManualDiagnosis(prevDiagnosis => ({
      ...prevDiagnosis,
      [eye]: {
        ...(prevDiagnosis?.[eye] || {}),
        [disease]: !(prevDiagnosis?.[eye]?.[disease])
      }
    }));
    setHasUnsavedChanges(true);
  };

  // Handler for custom disease input
  const handleCustomDiseaseChange = (eye, value) => {
    setCustomDiseases(prev => ({
      ...prev,
      [eye]: value
    }));
    setHasUnsavedChanges(true);
  };

  // Handler for diagnosis notes
  const handleDiagnosisNotesChange = (value) => {
    setDiagnosisNotes(value);
    setHasUnsavedChanges(true);
  };

  const handleReselectImages = async (newSelectedIds) => {
    setSelectedDisplayImages(newSelectedIds);
    setHasUnsavedChanges(true);
    try {
      const payload = {
        patient_id: patientData?.patient_id,
        selected_image_ids: newSelectedIds,
        exam_date: currentExamDate || null,
      };
      const resp = await fetch(`${backendUrl}/api/update_selection`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        console.warn('Update selection failed:', resp.status);
        return;
      }
      const updated = await resp.json();
      setPatientData(prev => prev ? ({
        ...prev,
        prediction_results: updated.prediction_results || prev.prediction_results,
        diagnosis_results: updated.diagnosis_results || prev.diagnosis_results,
        prediction_thresholds: updated.prediction_thresholds || prev.prediction_thresholds,
      }) : prev);
    } catch (e) {
      console.warn('Failed updating selection:', e);
    }
  };

  const handleDiscardChanges = () => {
    setSubmitMessage(''); // Clear any previous submit messages
    setHasUnsavedChanges(false);
    if (currentExamId) {
      fetchPatientData(currentExamId); // Reload original data for the current examination
    }
  };

  // Submit both diagnosis and image info in one request
  const handleSubmitDiagnosis = async () => {
    setIsSubmitting(true);
    setSubmitMessage('Submitting...');

    // Find the original patient data to compare against
    const originalPatientData = patientData.original;

    // 1. Prepare image info updates (only if they have changed)
    const imageUpdates = patientData.eye_images
      .map(currentImg => {
        const originalImg = originalPatientData.eye_images.find(img => img.id === currentImg.id);
        if (originalImg && (currentImg.type !== originalImg.type || currentImg.quality !== originalImg.quality)) {
          return { id: currentImg.id, type: currentImg.type, quality: currentImg.quality };
        }
        return null;
      })
      .filter(Boolean); // Remove nulls

    // 2. Prepare manual diagnosis data
    const manualDiagnosisPayload = {
      manual_diagnosis: manualDiagnosis,
      custom_diseases: customDiseases,
      diagnosis_notes: diagnosisNotes
    };

    // Check if there's manual diagnosis data or image updates to submit
    const hasManualDiagnosisData = Object.values(manualDiagnosis.left_eye).some(Boolean) || 
                                   Object.values(manualDiagnosis.right_eye).some(Boolean) ||
                                   customDiseases.left_eye || customDiseases.right_eye || 
                                   diagnosisNotes.trim();

    if (imageUpdates.length === 0 && !hasManualDiagnosisData) {
        setSubmitMessage("No changes to submit.");
        setTimeout(() => setSubmitMessage(''), 3000);
        setIsSubmitting(false);
        return;
    }

    const payload = {
      patient_id: patientData.patient_id,
      image_updates: imageUpdates.length > 0 ? imageUpdates : null,
      ...manualDiagnosisPayload,
    };

    try {
      // Use the dynamic backendUrl
      const response = await fetch(`${backendUrl}/api/submit_diagnosis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setSubmitMessage(result.status || 'Submission successful!');

      // Successfully submitted, so no more unsaved changes
      setHasUnsavedChanges(false);

      // Fetch the latest data from the server to ensure UI is in sync
      // This also updates the 'original' state for future comparisons
      if (currentExamId) {
        fetchPatientData(currentExamId);
      }

    } catch (error) {
      console.error("Failed to submit diagnosis:", error);
      setSubmitMessage(`Error: ${error.message}`);
    } finally {
      setTimeout(() => setSubmitMessage(''), 5000); // Keep message visible longer on error
      setIsSubmitting(false);
    }
  };

  const getDisplayedImageInfo = (imageId) => {
    // Look up the image info from the current patientData.eye_images
    // This allows the dropdowns to reflect local state changes
    return patientData?.eye_images.find(img => img.id === imageId);
  };

  // Handler to open the expanded image modal
  const openExpandedImageModal = (imageInfo) => {
    setExpandedImageInfo(imageInfo);
    setIsExpandedImageModalOpen(true);
  };

  // Handler to close the expanded image modal
  const closeExpandedImageModal = () => {
    setIsExpandedImageModalOpen(false);
    setExpandedImageInfo(null);
  };

  // Navigation function (manual go-to only)
  const navigateToExam = (examId) => {
    if (examId) {
      window.location.href = `${window.location.pathname}?ris_exam_id=${examId}`;
    }
  };

  // // 根据索引选择特定问诊记录
  // const selectConsultationByIndex = async (index, useRefined = true) => {
  //   try {
  //     const response = await fetch(`/api/consultation/${currentPatientId}/by_index/${index}?use_refined=${useRefined}`);
  //     const data = await response.json();
      
  //     if (data.status === 'success_refined' || data.status === 'success_original') {
  //       setConsultationData(data.consultation_data);
  //       setShowConsultationSelector(false);
  //       setSameNameConsultations([]);
  //     }
  //   } catch (error) {
  //     console.error('Failed to fetch consultation by index:', error);
  //   }
  // };
  
  
  // Handle consultation data changes
  const handleConsultationChange = useCallback((newData) => {
    setConsultationDataEdited(newData);
  }, []);
  
  // Submit consultation changes
  // const handleConsultationSubmit = useCallback(async () => {
  //   if (!consultationDataEdited || !currentExamId) return;
    
  //   setIsConsultationSubmitting(true);
  //   setConsultationSubmitMessage('');
    
  //   try {
  //     const response = await fetch(`${backendUrl}/api/consultation`, {
  //       method: 'POST',
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //       body: JSON.stringify({
  //         patient_id: currentExamId,
  //         consultation_data: consultationDataEdited
  //       }),
  //     });
      
  //     if (!response.ok) {
  //       throw new Error(`HTTP error! status: ${response.status}`);
  //     }
      
  //     const result = await response.json();
  //     setConsultationSubmitMessage(result.status || 'Consultation data saved!');
  //     setConsultationData(consultationDataEdited); // Update the original data
      
  //   } catch (e) {
  //     console.error("Failed to submit consultation data:", e);
  //     setConsultationSubmitMessage(`Error: ${e.message}`);
  //   } finally {
  //     setIsConsultationSubmitting(false);
  //     setTimeout(() => setConsultationSubmitMessage(''), 3000);
  //   }
  // }, [backendUrl, currentExamId, consultationDataEdited]);
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 p-6 font-inter text-gray-800 antialiased">
      {/* Header */}
      <header className="bg-white rounded-xl shadow-lg p-4 mb-6 flex items-center justify-center">
        <div className="flex items-center"> {/* Container for logo and title */}
          {/* Replaced the div with an img tag for the logo */}
          <img
            src="/TsinghuaLogo.svg" // Changed to /TsinghuaLogo.svg
            alt="Tsinghua Logo"
            className="w-14 h-14 rounded-full mr-4 shadow-inner object-cover" // Maintain circular and sizing
            onError={(e) => { // Optional: Add an error handler for the image
              e.target.onerror = null; // Prevent infinite loop if fallback also fails
              e.target.src = "https://placehold.co/56x56/507BCE/FFFFFF?text=Logo"; // Fallback placeholder
              console.warn("Local logo not found, falling back to placeholder.");
            }}
          />
          <h1 className="text-xl md:text-2xl font-extrabold text-gray-900 tracking-tight whitespace-nowrap">
            AI Eye Clinic 辅助诊断系统
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="bg-white rounded-xl shadow-xl p-8">
        {loading && (
          <p className="text-center text-blue-600 text-xl py-12">Loading patient data...</p>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-lg relative text-center mb-8" role="alert">
            <strong className="font-bold">Error:</strong>
            <span className="block sm:inline ml-2">{error}</span>
            {!currentExamId && (
              <div className="mt-2 text-sm">
                <p>Example URL: <code>{window.location.origin}{window.location.pathname}?ris_exam_id=12345</code></p>
              </div>
            )}
          </div>
        )}

        {patientData && (
          <div>
            {/* Patient ID and Navigation */}
            <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-2">
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-500">
                  {patientIdPrefix}<span className="text-blue-700 font-bold">{currentExamId || 'No ID'}</span>
                </span>
                
                {/* Exam Date Switcher - shown only when multiple instances available */}
                {availableExamInstances.length > 1 && (
                  <div className="flex items-center gap-2 ml-4 px-3 py-1 bg-yellow-50 border border-yellow-300 rounded-md">
                    <span className="text-xs font-medium text-gray-600">检查日期:</span>
                    <select
                      value={currentExamDate || ''}
                      onChange={(e) => handleExamDateSwitch(e.target.value)}
                      className="text-sm border border-gray-300 rounded px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      disabled={isLoadingInstances}
                    >
                      {availableExamInstances.map((instance) => (
                        <option key={instance.exam_date} value={instance.exam_date}>
                          {instance.exam_date} {instance.exam_date === availableExamInstances[0].exam_date ? '(最新)' : ''}
                        </option>
                      ))}
                    </select>
                    <span className="text-xs text-gray-500">
                      共 {availableExamInstances.length} 次检查
                    </span>
                  </div>
                )}
                
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    placeholder="Go to exam ID..."
                    className="px-2 py-1 text-sm border border-gray-300 rounded"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && e.target.value.trim()) {
                        navigateToExam(e.target.value.trim());
                      }
                    }}
                  />
                  <button
                    onClick={(e) => {
                      const input = e.target.previousElementSibling;
                      if (input.value.trim()) {
                        navigateToExam(input.value.trim());
                      }
                    }}
                    className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                  >
                    Go
                  </button>
                </div>
              </div>
              {/* Removed redundant top reselect button */}
              {/* ...existing code... */}
            </div>

            {/* New Left-Right Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10 items-start">
              {/* Left column: Consultation Info - Add max-height and overflow */}
              <div className="lg:col-span-1">
              {/* 患者姓名搜索（可选） */}
              <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  按患者姓名搜索问诊信息
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={patientNameSearch}
                    onChange={(e) => handlePatientNameSearch(e.target.value)}
                    onFocus={() => patientNameSearch && setShowNameSuggestions(true)}
                    placeholder="输入患者姓名..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {showNameSuggestions && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                      {availablePatientNames
                        .filter(name => name.includes(patientNameSearch))
                        .map((name, idx) => (
                          <div
                            key={idx}
                            onClick={() => selectPatientName(name)}
                            className="px-3 py-2 cursor-pointer hover:bg-blue-50 border-b border-gray-100 last:border-b-0"
                          >
                            {name}
                          </div>
                        ))}
                    </div>
                  )}
                </div>
                {patientNameSearch && (
                  <div className="mt-2 flex space-x-2">
                    <button
                      onClick={() => fetchConsultationInfo(currentPatientId, patientNameSearch)}
                      className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                    >
                      搜索
                    </button>
                    <button
                      onClick={() => { setPatientNameSearch(''); fetchConsultationInfo(currentPatientId); }}
                      className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700"
                    >
                      重置
                    </button>
                  </div>
                )}
              </div>

              {/* 原有问诊信息表单 - Add max-height with overflow */}
              <div className="max-h-[600px] overflow-y-auto">
                <ConsultationInfoSection 
                  consultationData={consultationDataEdited}
                  onChange={handleConsultationChange}
                  onSubmit={handleConsultationSubmit}
                  isSubmitting={isConsultationSubmitting}
                />
                {consultationSubmitMessage && (
                  <div className="mt-2 text-center">
                    <p className="text-green-600 text-sm">{consultationSubmitMessage}</p>
                  </div>
                )}
              </div>
            </div>
              
              {/* Right column: Images */}
              <div className="lg:col-span-2 h-full">
                {/* Image Display Section */}
                <div className="relative flex h-full items-center justify-center bg-gray-50 p-6 rounded-xl shadow-inner min-h-[560px]">
                  <div className="flex flex-wrap justify-center gap-6 mx-auto">
                    {selectedDisplayImages.map((imageId, index) => {
                      const imgInfo = getDisplayedImageInfo(imageId);
                      return (
                        <div
                          key={index}
                          className="flex-shrink-0 w-56 h-64 bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 transform hover:scale-105 transition-transform duration-200 group cursor-pointer"
                          onClick={() => imgInfo && openExpandedImageModal(imgInfo)}
                        >
                          {imgInfo ? (
                            <>
                              <img
                                src={`data:image/png;base64,${imgInfo.base64_data}`}
                                alt={`Image ${index + 1}`}
                                className="w-full h-48 object-cover border-b border-gray-200 group-hover:opacity-80 transition-opacity"
                              />
                              <div className="p-3 text-sm text-center">
                                <p className="font-semibold text-gray-900 group-hover:text-blue-700 transition-colors">IMAGE {index + 1}</p>
                                <p className="text-gray-600">{imgInfo.type}</p>
                              </div>
                            </>
                          ) : (
                            <div className="w-full h-64 bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
                              No Image Selected
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  {/* Bottom-right overlay reselect button (single source of truth) */}
                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="absolute bottom-3 right-3 z-20 px-3 py-2 rounded-full
                               bg-white/90 text-blue-700 border border-blue-200 shadow-lg backdrop-blur-sm
                               hover:bg-white hover:shadow-xl transition-all duration-200 ease-in-out focus:outline-none
                               focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                    title="重新选择图片 (Reselect Images)"
                    aria-label="重新选择图片 (Reselect Images)"
                  >
                    重新选择图片
                  </button>
                </div>
              </div>
            </div>

            {/* Replace the existing AI sections and LLM Chat layout with new grid structure */}
            <div className="mb-6">
              {/* Side-by-side AI sections - Reduced height */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Left: AI Examination Summary - Center content, reduce padding */}
                <div className="p-4 rounded-2xl shadow-sm border border-indigo-300 bg-indigo-50/60">
                  <h3 className="text-xl font-bold mb-3 text-indigo-900 text-center tracking-tight">
                    AI检查摘要 (AI Examination Summary)
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {['left_eye','right_eye'].map((eyeKey) => {
                      const eyeLabel = eyeKey === 'left_eye' ? '左眼 (Left Eye)' : '右眼 (Right Eye)';
                      const { primaries, secondaries } = getEyeHighlights(eyeKey);
                      const statusTextMap = {
                        '明显偏高': '明显高于阈值 / Markedly above T',
                        '较高': '高于阈值 / Above T',
                        '接近阈值': '接近阈值 / Near T',
                        '较低': '低于阈值 / Below T',
                      };
                      return (
                        <div key={eyeKey} className="p-3 rounded-xl bg-white border border-indigo-200 shadow-sm">
                          <div className="text-sm font-semibold text-gray-800 mb-2 text-center">{eyeLabel}</div>
                          {primaries.length > 0 ? (
                            <div className="text-center">
                              {/* Primaries: show all above-threshold or the top one if none above */}
                              <div className="flex flex-col gap-2">
                                {primaries.map((pItem) => (
                                  pItem.key === '正常' ? (
                                    <div key={pItem.key}>
                                      <div className="mt-1 text-xs text-gray-600">总体判断 (Overall)</div>
                                      <div className="text-lg md:text-xl font-semibold text-green-700">正常 (Normal)</div>
                                    </div>
                                  ) : (
                                    <div key={pItem.key}>
                                      <div className="text-xs text-gray-600">首要考虑 (Primary)</div>
                                      <div className="text-base md:text-lg font-semibold text-gray-900">{pItem.name}</div>
                                      <div className="flex items-center justify-center gap-2 mt-1">
                                        <span className="inline-flex items-center px-2 py-0.5 rounded bg-blue-50 text-blue-800 text-xs border border-blue-200">
                                          P {formatProb(pItem.p)} · T {formatProb(pItem.t)}
                                        </span>
                                        <span className={
                                          `inline-flex items-center px-2 py-0.5 rounded text-xs border ${
                                            pItem.status === '明显偏高' ? 'bg-red-50 text-red-800 border-red-200' :
                                            pItem.status === '较高' ? 'bg-orange-50 text-orange-800 border-orange-200' :
                                            pItem.status === '接近阈值' ? 'bg-yellow-50 text-yellow-800 border-yellow-200' :
                                            'bg-gray-50 text-gray-700 border-gray-200'
                                          }`
                                        }>
                                          {statusTextMap[pItem.status] || pItem.status}
                                        </span>
                                      </div>
                                    </div>
                                  )
                                ))}
                              </div>

                              {secondaries.length > 0 && (
                                <div className="mt-2">
                                  <div className="text-xs font-medium text-gray-500 tracking-wide">次要关注 (Secondary)</div>
                                  <div className="mt-1 flex flex-wrap justify-center gap-1">
                                    {secondaries.map((o) => (
                                      <span
                                        key={o.key}
                                        className="px-2 py-0.5 rounded-full bg-amber-400/10 text-amber-700 text-xs border border-amber-300/40 opacity-85"
                                     >
                                        {o.name}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="text-sm text-gray-500 text-center">暂无要点 (No highlights)</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Right: Prediction Probability Bars (Further shrunk) */}
                <div className="p-3 rounded-lg shadow-sm border border-gray-100 bg-white">
                  <h3 className="text-lg font-semibold mb-2 text-gray-700 text-center">AI预测概率 (Model Prediction Probabilities)</h3>
                  <p className="text-xs text-gray-500 mb-2 text-center">彩条长度按阈值重新映射: 阈值位于条形中点 (50%)。</p>
                  <div className="overflow-y-auto max-h-[320px]">
                    <table className="min-w-full table-fixed text-xs">
                      <thead className="sticky top-0 bg-white shadow-sm z-10">
                        <tr>
                          <th className="w-[30%] p-1 text-left text-gray-600 font-medium">疾病 (Disease)</th>
                          <th className="p-1 text-center text-gray-600 font-medium">左眼 (L)</th>
                          <th className="p-1 text-center text-gray-600 font-medium">右眼 (R)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {diseaseOrder.map(dk => {
                          const thresholds = patientData?.prediction_thresholds || {};
                          const t = thresholds[dk] ?? 0.5;
                          const leftProb = patientData?.prediction_results?.left_eye?.[dk] ?? 0.0;
                          const rightProb = patientData?.prediction_results?.right_eye?.[dk] ?? 0.0;
                          const leftWidthRaw = mapProbToWidth(leftProb, t) * 100;
                          const rightWidthRaw = mapProbToWidth(rightProb, t) * 100;
                          const clamp = (v) => Math.min(100, Math.max(0, v));
                          const leftWidth = clamp(leftWidthRaw);
                          const rightWidth = clamp(rightWidthRaw);
                          return (
                            <tr key={dk} className="border-t border-gray-100">
                              <td className="p-1 align-middle text-gray-700 whitespace-nowrap font-medium">
                                <div className="flex flex-col">
                                  <span className="font-semibold text-gray-800">{diseaseInfo[dk].chinese}</span>
                                  <span className="text-xs text-gray-500">{diseaseInfo[dk].shortName}</span>
                                </div>
                              </td>
                              <td className="p-1">
                                <div className="relative h-3 rounded-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-400 overflow-hidden shadow-inner">
                                  <div className="absolute top-0 left-0 h-full bg-green-600/20" style={{ width: `${leftWidth}%` }} />
                                  <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gray-600/70" />
                                  <div className="absolute top-0 h-full flex items-center" style={{ left: `${leftWidth}%`, transform: 'translateX(-50%)' }}
                                       title={`${diseaseInfo[dk].fullName}: P:${formatProb(leftProb)} T:${formatProb(t)}`}>
                                    <div className="w-0.5 h-full bg-blue-700/70"></div>
                                    <div className="absolute left-1/2 top-1/2 w-1.5 h-1.5 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-blue-600 border border-white shadow-sm" />
                                  </div>
                                  <div className="absolute inset-0 flex justify-between px-1 text-[8px] leading-3 text-gray-600 select-none font-medium">
                                    <span>{formatProb(leftProb)}</span>
                                    <span className="text-gray-500">T:{formatProb(t)}</span>
                                  </div>
                                </div>
                              </td>
                              <td className="p-1">
                                <div className="relative h-3 rounded-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-400 overflow-hidden shadow-inner">
                                  <div className="absolute top-0 left-0 h-full bg-green-600/20" style={{ width: `${rightWidth}%` }} />
                                  <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gray-600/70" />
                                  <div className="absolute top-0 h-full flex items-center" style={{ left: `${rightWidth}%`, transform: 'translateX(-50%)' }}
                                       title={`${diseaseInfo[dk].fullName}: P:${formatProb(rightProb)} T:${formatProb(t)}`}>
                                    <div className="w-0.5 h-full bg-blue-700/70"></div>
                                    <div className="absolute left-1/2 top-1/2 w-1.5 h-1.5 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-blue-600 border border-white shadow-sm" />
                                  </div>
                                  <div className="absolute inset-0 flex justify-between px-1 text-[8px] leading-3 text-gray-600 select-none font-medium">
                                    <span>{formatProb(rightProb)}</span>
                                    <span className="text-gray-500">T:{formatProb(t)}</span>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              {/* Full-width LLM Chat section below - unchanged */}
              <div className="p-5 rounded-lg shadow-sm border border-gray-100 bg-white w-full">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-lg font-semibold text-gray-800">临床助手 / LLM Chat</h3>
                  {!llmLoading ? (
                    <button
                      className="px-3 py-1.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-700"
                      onClick={regenerateSideOpinion}
                      title="根据最新结果生成摘要"
                    >
                      更新最新结果
                    </button>
                  ) : (
                    <button
                      className="px-3 py-1.5 text-xs rounded bg-red-600 text-white hover:bg-red-700"
                      onClick={() => llmAbortCtrl?.abort()}
                      title="停止生成"
                    >
                      停止
                    </button>
                  )}
                </div>

                {/* Chat scroll area with fixed height - Made 2x taller */}
                <div
                  ref={sideChatScrollRef}
                  onScroll={handleSideChatScroll}
                  className="relative h-[600px] overflow-y-auto overscroll-contain
                           border border-gray-200 rounded p-2 bg-gray-50 mb-3"
                >
                  {sideChatMessages.length === 0 && (
                    <div className="text-gray-400 text-sm">暂无对话</div>
                  )}
                  {sideChatMessages.map((m, i) => (
                    <div key={i} className={`mb-3 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
                      <div className={`inline-block px-3 py-2 rounded-lg max-w-[90%] ${
                        m.role === 'user'
                          ? 'bg-blue-100 text-blue-900'
                          : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                      }`}>
                        {m.role === 'user' ? (
                          <span className="text-sm whitespace-pre-wrap">{m.content}</span>
                        ) : (
                          <div className="text-sm">
                            <div
                              className="prose prose-sm max-w-none prose-headings:mt-2 prose-headings:mb-1 prose-p:my-1"
                              dangerouslySetInnerHTML={{ __html: renderMarkdownContent(m.content) }}
                            />
                            {llmLoading && i === sideChatMessages.length - 1 && (
                              <span className="inline-block w-2 h-4 bg-gray-400 ml-1 animate-pulse" />
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* "Back to bottom" button */}
                {!autoScrollChat && (
                  <button
                    onClick={() => {
                      const el = sideChatScrollRef.current;
                      if (!el) return;
                      el.scrollTop = el.scrollHeight;
                      setAutoScrollChat(true);
                    }}
                    className="absolute bottom-[60px] right-8 px-2 py-1 text-xs rounded
                             bg-blue-600 text-white shadow hover:bg-blue-700 z-20"
                    title="回到底部"
                  >
                    回到底部
                  </button>
                )}

                {/* Input area */}
                <div className="flex gap-2">
                  <input
                    className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
                    placeholder="在此输入问题…"
                    value={sideChatInput}
                    onChange={(e) => setSideChatInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !llmLoading) sendSideChatStreaming(); }}
                    disabled={llmLoading}
                  />
                  {!llmLoading ? (
                    <button
                      className="px-3 py-2 rounded bg-blue-600 text-white text-sm hover:bg-blue-700"
                      onClick={() => sendSideChatStreaming()}
                      disabled={!sideChatInput.trim()}
                    >
                      发送
                    </button>
                  ) : (
                    <button
                      className="px-3 py-2 rounded bg-red-600 text-white text-sm hover:bg-red-700"
                      onClick={() => llmAbortCtrl?.abort()}
                    >
                      停止
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Unified Interactive Correction Section */}
            <div className="mb-10 p-6 rounded-lg shadow-sm border border-gray-200 bg-white">
              <h3 className="text-xl font-semibold mb-2 text-gray-800 text-center">人工复检区 (Re-examination)</h3>
              <p className="text-xs text-gray-500 mb-5 text-center">在此对影像类型/质量与疾病诊断结果进行人工复核与修改 (Review & adjust image metadata and disease diagnoses)</p>
              {/* Image Condition Row */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 tracking-wide">影像情况 (Image Condition)</h4>
                <div className="flex flex-col md:flex-row justify-center items-stretch gap-x-4 gap-y-4 overflow-x-auto pb-2">
                  {selectedDisplayImages.map((imageId, imgIndex) => {
                    const imgInfo = getDisplayedImageInfo(imageId);
                    if (!imgInfo) return null;
                    const typeOptions = ['--- Select ---', '左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'];
                    const defaultQualityOption = '图像质量可用';
                    const qualityOptions = [defaultQualityOption, '图像质量高', '图像质量差', '--- Select ---'];
                    return (
                      <div key={imgIndex} className="flex flex-col items-center gap-2 p-3 bg-gray-50 rounded-md border border-gray-200 flex-grow basis-0 min-w-[200px]">
                        <span className="font-medium text-gray-700 text-sm whitespace-nowrap">Image {imgIndex + 1}</span>
                        <div className="flex flex-row gap-2 w-full">
                          <select
                            value={imgInfo.type}
                            onChange={(e) => handleImageDetailChange(imgInfo.id, 'type', e.target.value)}
                            className="p-2 border border-gray-300 rounded-md bg-white text-gray-700 text-xs md:text-sm flex-grow focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-150 shadow-sm"
                          >
                            {typeOptions.map(option => (
                              <option key={option} value={option === '--- Select ---' ? '' : option}>{option}</option>
                            ))}
                          </select>
                          <select
                            value={imgInfo.quality || defaultQualityOption}
                            onChange={(e) => handleImageDetailChange(imgInfo.id, 'quality', e.target.value)}
                            className="p-2 border border-gray-300 rounded-md bg-white text-gray-700 text-xs md:text-sm flex-grow focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-150 shadow-sm"
                          >
                            {qualityOptions.map(option => (
                              <option key={option} value={option === '--- Select ---' ? '' : option}>{option}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              {/* Manual Diagnosis Section */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 tracking-wide">手动诊断 (Manual Disease Diagnosis)</h4>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {/* Left Eye Manual Diagnosis */}
                  <div className="border border-gray-200 rounded-lg p-4 bg-blue-50">
                    <h5 className="font-medium text-gray-700 mb-3 text-center">左眼 (Left Eye)</h5>
                    <div className="space-y-2">
                      {Object.entries(manualDiseaseInfo).map(([disease, info]) => (
                        <label key={disease} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={(manualDiagnosis?.left_eye?.[disease]) || false}
                            onChange={() => handleManualDiagnosisToggle('left_eye', disease)}
                            className="form-checkbox h-4 w-4 text-blue-600"
                          />
                          <span className="text-sm">
                            {info.chinese} / {info.english}
                          </span>
                        </label>
                      ))}
                      
                      <div className="mt-3">
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          其他疾病 (Custom Disease):
                        </label>
                        <input
                          type="text"
                          value={customDiseases?.left_eye || ''}
                          onChange={(e) => handleCustomDiseaseChange('left_eye', e.target.value)}
                          placeholder="输入其他疾病 / Enter custom disease"
                          className="w-full px-2 py-1 border border-gray-300 rounded-md text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  </div>
                  
                  {/* Right Eye Manual Diagnosis */}
                  <div className="border border-gray-200 rounded-lg p-4 bg-green-50">
                    <h5 className="font-medium text-gray-700 mb-3 text-center">右眼 (Right Eye)</h5>
                    <div className="space-y-2">
                      {Object.entries(manualDiseaseInfo).map(([disease, info]) => (
                        <label key={disease} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={(manualDiagnosis?.right_eye?.[disease]) || false}
                            onChange={() => handleManualDiagnosisToggle('right_eye', disease)}
                            className="form-checkbox h-4 w-4 text-green-600"
                          />
                          <span className="text-sm">
                            {info.chinese} / {info.english}
                          </span>
                        </label>
                      ))}
                      
                      <div className="mt-3">
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          其他疾病 (Custom Disease):
                        </label>
                        <input
                          type="text"
                          value={customDiseases?.right_eye || ''}
                          onChange={(e) => handleCustomDiseaseChange('right_eye', e.target.value)}
                          placeholder="输入其他疾病 / Enter custom disease"
                          className="w-full px-2 py-1 border border-gray-300 rounded-md text-xs focus:outline-none focus:ring-1 focus:ring-green-500"
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Diagnosis Notes Section */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    诊断说明 (Diagnosis Notes):
                  </label>
                  <textarea
                    value={diagnosisNotes}
                    onChange={(e) => handleDiagnosisNotesChange(e.target.value)}
                    placeholder="输入详细的诊断说明和观察结果 / Enter detailed diagnosis notes and observations"
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                  />
                </div>
              </div>


              {/* Action Buttons Inside Unified Section */}
              <div className="mt-8 flex flex-col md:flex-row items-center md:items-start justify-between gap-4">
                <div className="w-full md:flex-1 text-sm font-medium text-green-700 min-h-[1.5rem] flex items-center" aria-live="polite">
                  <span className={`${submitMessage ? 'opacity-100' : 'opacity-0'} transition-opacity duration-200`}>
                    {submitMessage || ''}
                  </span>
                </div>
                <div className="flex gap-4 md:justify-end w-full md:w-auto">
                  <button
                    onClick={handleAlterThreshold}
                    disabled={isAlteringThreshold}
                    className={`px-6 py-2.5 rounded-lg shadow-md transition-colors duration-200 text-sm font-semibold
                      ${isAlteringThreshold ? 'bg-blue-300 cursor-not-allowed text-white' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
                  >
                    {isAlteringThreshold ? '更新中...' : `阈值套装 ${activeThresholdSet + 1} (Alter Threshold)`}
                  </button>
                  <button
                    onClick={handleDiscardChanges}
                    className="px-6 py-2.5 bg-red-500 text-white rounded-lg shadow-md hover:bg-red-600 transition-colors duration-200 text-sm font-semibold"
                  >
                    放弃 (Discard)
                  </button>
                  <button
                    onClick={handleSubmitDiagnosis}
                    disabled={isSubmitting}
                    className={`px-6 py-2.5 rounded-lg shadow-md transition-colors duration-200 text-sm font-semibold
                      ${isSubmitting ? 'bg-green-300 cursor-not-allowed text-white' : 'bg-green-600 text-white hover:bg-green-700'}`}
                  >
                    {isSubmitting ? '提交中...' : '提交 (Submit)'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        {showConsultationSelector && sameNameConsultations.length > 0 && (
  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
    <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-900">
            选择问诊记录 - {sameNameConsultations[0]?.name || ''}
          </h2>
          <button
            onClick={() => setShowConsultationSelector(false)}
            className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        <div className="space-y-3">
          {sameNameConsultations.map((c, idx) => (
            <div
              key={idx}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
              onClick={() => selectConsultationByIndex(c.index, c.isRefined)}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div><span className="font-medium text-gray-600">年龄：</span><span className="text-gray-900">{c.age || '-'}</span></div>
                    <div><span className="font-medium text-gray-600">性别：</span><span className="text-gray-900">{c.gender || '-'}</span></div>
                    <div><span className="font-medium text-gray-600">电话：</span><span className="text-gray-900">{c.phone || '-'}</span></div>
                    <div><span className="font-medium text-gray-600">提交时间：</span><span className="text-gray-900">{c.submissionTime ? new Date(c.submissionTime).toLocaleString('zh-CN') : '-'}</span></div>
                  </div>
                </div>
                <div className="ml-4 flex flex-col items-end space-y-1">
                  {c.isRefined && <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">修订后</span>}
                  {c.hasRefined && !c.isRefined && <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">原始版本</span>}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={() => setShowConsultationSelector(false)}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            取消
          </button>
        </div>
      </div>
    </div>
  </div>
)}
      </main>

      {/* Disclaimer */}
      <footer className="mt-8 text-center text-gray-600 text-xs">
        *人工智能系统存在一定局限性，可能产生误差，相关检测结果仅供参考，不构成最终决策依据。
        <br />
        (*AI system has certain limitations and may produce errors. Related detection results are for reference only, not as final decision basis.)
      </footer>

      <ReselectImageModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        allImages={patientData?.eye_images || []}
        onSelectImages={handleReselectImages}
        selectedImageIds={selectedDisplayImages}
      />

      {/* New Expanded Image Modal */}
      <ExpandedImageModal
        isOpen={isExpandedImageModalOpen}
        onClose={closeExpandedImageModal}
        imageInfo={expandedImageInfo}
      />
    </div>
  );
}

export default App;
