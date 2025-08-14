import React, { useState, useEffect, useCallback } from 'react';

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

  // Get ris_exam_id from URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const currentExamId = urlParams.get('ris_exam_id');

  // Dynamically determine the backend URL
  // Use the same hostname as the frontend, but on port 8000
  // Fallback to localhost for local development
  const backendHost = window.location.hostname;
  const backendUrl = `http://${backendHost}:8000`;

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
      chinese: '高度近视', 
      english: 'High Myopia',
      fullName: '高度近视 (High Myopia)',
      shortName: 'HM'
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
    其它视网膜病: { 
      chinese: '其它视网膜病', 
      english: 'Other Retinal Diseases',
      fullName: '其它视网膜病 (Other Retinal)',
      shortName: 'Other Retinal'
    },
    其它黄斑病变: { 
      chinese: '其它黄斑病变', 
      english: 'Other Macular Diseases',
      fullName: '其它黄斑病变 (Other Macular)',
      shortName: 'Other Macular'
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

  // Helper function to deep compare two objects
  const isDataEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  // Build a natural Chinese summary per eye; skip '正常' in secondary mentions
  const buildEyeSummary = (eyeKey) => {
    try {
      const preds = patientData?.prediction_results?.[eyeKey];
      const thresholds = patientData?.prediction_thresholds || {};
      if (!preds) return '';

      const eyeLabel = eyeKey === 'left_eye' ? '左眼' : '右眼';

      // Build array of { key, p, t }
      const items = diseaseOrder.map((dk) => ({
        key: dk,
        p: preds[dk] ?? 0,
        t: thresholds[dk] ?? 0.5,
      }));

      // Sort by probability desc
      items.sort((a, b) => b.p - a.p);
      const [top, ...rest] = items;
      if (!top) return '';

      const topName = diseaseInfo[top.key]?.chinese || top.key;

      // Choose a natural phrase based on relation to threshold
      const t = top.t ?? 0.5;
      const p = top.p ?? 0;
      let probPhrase = '概率较高';
      if (p >= t * 1.2) probPhrase = '概率明显偏高';
      else if (p >= t) probPhrase = '概率较高';
      else if (p >= t * 0.8) probPhrase = '概率接近阈值';
      else probPhrase = '概率较低';

      // If top is 正常, use a more natural normal-first sentence
      if (top.key === '正常') {
        const others = rest
          .filter((x) => x.key !== '正常' && (x.p ?? 0) >= 0.2 * (x.t ?? 0.5))
          .sort((a, b) => b.p - a.p)
          .slice(0, 2)
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
      const others = rest
        .filter((x) => x.key !== '正常' && (x.p ?? 0) >= 0.2 * (x.t ?? 0.5))
        .sort((a, b) => b.p - a.p)
        .slice(0, 2)
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

  const fetchPatientData = useCallback(async (examId) => {
    if (!examId) {
      setError('No ris_exam_id provided in URL. Please add ?ris_exam_id=<exam_id> to the URL.');
      setLoading(false);
      return;
    }

    console.log(`Starting to fetch patient data for exam ID: ${examId}`);
    const startTime = performance.now();
    setLoading(true);
    setError(null);

    try {
      console.log(`Fetching from: ${backendUrl}/api/patients/${examId}`);
      const response = await fetch(`${backendUrl}/api/patients/${examId}`);
      
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

      // Initialize the 4 display images
      const uniqueImageTypes = {};
      result.eye_images.forEach(img => {
        if (img.type && !uniqueImageTypes[img.type]) {
          uniqueImageTypes[img.type] = img.id;
        }
      });
      let initialSelection = Object.values(uniqueImageTypes);
      
      // Fill remaining slots if necessary
      const remainingImages = result.eye_images.filter(img => !initialSelection.includes(img.id));
      while (initialSelection.length < 4 && remainingImages.length > 0) {
        initialSelection.push(remainingImages.shift().id);
      }
      setSelectedDisplayImages(initialSelection.slice(0, 4));
      setHasUnsavedChanges(false); // Reset unsaved changes flag
      
      // Initialize manual diagnosis states
      const initialManualDiagnosis = {
        left_eye: Object.keys(manualDiseaseInfo).reduce((acc, key) => ({ ...acc, [key]: false }), {}),
        right_eye: Object.keys(manualDiseaseInfo).reduce((acc, key) => ({ ...acc, [key]: false }), {})
      };
      setManualDiagnosis(initialManualDiagnosis);
      setCustomDiseases({ left_eye: '', right_eye: '' });
      setDiagnosisNotes('');
      
      console.log(`Patient data processing completed for ${examId}`);
    } catch (e) {
      const errorTime = performance.now() - startTime;
      console.error(`Failed to fetch patient data after ${errorTime.toFixed(2)}ms:`, e);
      setError(`Failed to load examination data: ${e.message}. Please check if the examination ID is correct and the server is running.`);
    } finally {
      setLoading(false);
    }
  }, [backendUrl]);

  // Fetch data on component mount or when examId changes
  useEffect(() => {
    if (currentExamId) {
      fetchPatientData(currentExamId);
    } else {
      setError('No ris_exam_id provided in URL. Please add ?ris_exam_id=<exam_id> to the URL.');
      setLoading(false);
    }
  }, [currentExamId, fetchPatientData]);

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
        ...prevDiagnosis[eye],
        [disease]: !prevDiagnosis[eye][disease]
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

  const handleReselectImages = (newSelectedIds) => {
    setSelectedDisplayImages(newSelectedIds);
    setHasUnsavedChanges(true);
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

  // Navigation functions
  const navigateToExam = (examId) => {
    if (examId) {
      window.location.href = `${window.location.pathname}?ris_exam_id=${examId}`;
    }
  };

  const navigateToPrevious = () => {
    // You can implement this based on your exam ID logic
    // For now, this is a placeholder that you can customize
    const currentId = parseInt(currentExamId);
    if (!isNaN(currentId) && currentId > 1) {
      navigateToExam(currentId - 1);
    }
  };

  const navigateToNext = () => {
    // You can implement this based on your exam ID logic
    // For now, this is a placeholder that you can customize
    const currentId = parseInt(currentExamId);
    if (!isNaN(currentId)) {
      navigateToExam(currentId + 1);
    }
  };

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
              <button
                onClick={() => setIsModalOpen(true)}
                className="px-4 py-1.5 bg-green-600 text-white rounded-lg shadow-md
                hover:bg-green-700 transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 text-sm"
              >
                重新选择图片 (Reselect Image)
              </button>
            </div>

            {/* Image Display Section */}
            <div className="relative flex items-center justify-center mb-10 bg-gray-50 p-6 rounded-xl shadow-inner">
              {/* Left arrow for previous examination */}
              <IconButton onClick={navigateToPrevious} className="absolute left-3 z-10">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </IconButton>

              <div className="flex flex-wrap justify-center gap-6 mx-12">
                {selectedDisplayImages.map((imageId, index) => {
                  const imgInfo = getDisplayedImageInfo(imageId);
                  return (
                    <div
                      key={index}
                      className="flex-shrink-0 w-64 h-72 bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 transform hover:scale-105 transition-transform duration-200 group cursor-pointer" // Added cursor-pointer
                      onClick={() => imgInfo && openExpandedImageModal(imgInfo)} // Add onClick to open modal
                    >
                      {imgInfo ? (
                        <>
                          <img
                            src={`data:image/png;base64,${imgInfo.base64_data}`}
                            alt={`Image ${index + 1}`}
                            className="w-full h-56 object-cover border-b border-gray-200 group-hover:opacity-80 transition-opacity"
                          />
                          <div className="p-3 text-sm text-center">
                            <p className="font-semibold text-gray-900 group-hover:text-blue-700 transition-colors">IMAGE {index + 1}</p>
                            <p className="text-gray-600">{imgInfo.type}</p>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-72 bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
                          No Image Selected
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Right arrow for next examination */}
              <IconButton onClick={navigateToNext} className="absolute right-3 z-10">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </IconButton>
            </div>

            {/* Prediction Probability Bars Section */}
            <div className="mb-8 p-5 rounded-lg shadow-sm border border-gray-100 bg-white">
              <h3 className="text-lg font-semibold mb-3 text-gray-700 text-center">AI预测概率 (Model Prediction Probabilities)</h3>
              <p className="text-xs text-gray-500 mb-4 text-center">彩条长度按阈值重新映射: 阈值位于条形中点 (50%)，左侧表示低于阈值，右侧高于阈值。</p>
              <div className="overflow-x-auto">
                <table className="min-w-full table-fixed text-xs md:text-sm">
                  <thead>
                    <tr>
                      <th className="w-40 p-2 text-left text-gray-600 font-medium">疾病 (Disease)</th>
                      <th className="p-2 text-center text-gray-600 font-medium">左眼 (Left)</th>
                      <th className="p-2 text-center text-gray-600 font-medium">右眼 (Right)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diseaseOrder.map(dk => {
                      const thresholds = patientData?.prediction_thresholds || {};
                      const t = thresholds[dk] ?? 0.5; // fallback
                      const leftProb = patientData?.prediction_results?.left_eye?.[dk] ?? 0.0;
                      const rightProb = patientData?.prediction_results?.right_eye?.[dk] ?? 0.0;
                      const leftWidthRaw = mapProbToWidth(leftProb, t) * 100;
                      const rightWidthRaw = mapProbToWidth(rightProb, t) * 100;
                      const clamp = (v) => Math.min(100, Math.max(0, v));
                      const leftWidth = clamp(leftWidthRaw);
                      const rightWidth = clamp(rightWidthRaw);
                      return (
                        <tr key={dk} className="border-t border-gray-100">
                          <td className="p-2 align-middle text-gray-700 whitespace-nowrap font-medium">
                            <div className="flex flex-col">
                              <span className="text-sm font-semibold text-gray-800">{diseaseInfo[dk].chinese}</span>
                              <span className="text-xs text-gray-500">{diseaseInfo[dk].english}</span>
                            </div>
                          </td>
                          {/* Left eye bar */}
                          <td className="p-2">
                            <div className="relative h-5 rounded-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-400 overflow-hidden shadow-inner">
                              {/* Fill overlay */}
                              <div className="absolute top-0 left-0 h-full bg-green-600/20" style={{ width: `${leftWidth}%` }} />
                              {/* Threshold marker at 50% */}
                              <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gray-600/70" />
                              {/* Probability marker (line + diamond) */}
                              <div className="absolute top-0 h-full flex items-center" style={{ left: `${leftWidth}%`, transform: 'translateX(-50%)' }}
                                   title={`${diseaseInfo[dk].fullName}: P:${formatProb(leftProb)} T:${formatProb(t)}`}
                                   aria-label={`Left eye ${diseaseInfo[dk].fullName} probability ${formatProb(leftProb)}, threshold ${formatProb(t)}`}>
                                <div className="w-0.5 h-full bg-blue-700/70"></div>
                                <div className="absolute left-1/2 top-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-blue-600 border border-white shadow-sm" />
                              </div>
                              {/* Labels */}
                              <div className="absolute inset-0 flex justify-between px-1 text-[10px] leading-5 text-gray-600 select-none font-medium">
                                <span>{formatProb(leftProb)}</span>
                                <span className="text-gray-500">T:{formatProb(t)}</span>
                              </div>
                            </div>
                          </td>
                          {/* Right eye bar */}
                          <td className="p-2">
                            <div className="relative h-5 rounded-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-400 overflow-hidden shadow-inner">
                              <div className="absolute top-0 left-0 h-full bg-green-600/20" style={{ width: `${rightWidth}%` }} />
                              <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gray-600/70" />
                              <div className="absolute top-0 h-full flex items-center" style={{ left: `${rightWidth}%`, transform: 'translateX(-50%)' }}
                                   title={`${diseaseInfo[dk].fullName}: P:${formatProb(rightProb)} T:${formatProb(t)}`}
                                   aria-label={`Right eye ${diseaseInfo[dk].fullName} probability ${formatProb(rightProb)}, threshold ${formatProb(t)}`}>
                                <div className="w-0.5 h-full bg-blue-700/70"></div>
                                <div className="absolute left-1/2 top-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-blue-600 border border-white shadow-sm" />
                              </div>
                              <div className="absolute inset-0 flex justify-between px-1 text-[10px] leading-5 text-gray-600 select-none font-medium">
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

            {/* AI Summary Section */}
            <div className="mb-8 p-5 rounded-lg shadow-sm border border-gray-100 bg-white">
              <h3 className="text-lg font-semibold mb-3 text-gray-700 text-center">AI总结 (AI Detection Summary)</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-800">
                <div className="p-3 rounded-md bg-gray-50 border border-gray-100">
                  <div className="font-semibold mb-1">左眼 (Left Eye)</div>
                  <p className="leading-6">{buildEyeSummary('left_eye') || '暂无总结'}</p>
                </div>
                <div className="p-3 rounded-md bg-gray-50 border border-gray-100">
                  <div className="font-semibold mb-1">右眼 (Right Eye)</div>
                  <p className="leading-6">{buildEyeSummary('right_eye') || '暂无总结'}</p>
                </div>
              </div>
            </div>

            {/* Unified Interactive Correction Section */}
            <div className="mb-10 p-6 rounded-lg shadow-sm border border-gray-200 bg-white">
              <h3 className="text-xl font-semibold mb-2 text-gray-800 text-center">人工校正区 (Interactive Human Correction)</h3>
              <p className="text-xs text-gray-500 mb-5 text-center">在此对影像类型/质量与疾病诊断结果进行人工复核与修改 (Review & adjust image metadata and disease diagnoses)</p>
              {/* Image Condition Row */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-gray-700 mb-3 tracking-wide">影像情况 (Image Condition)</h4>
                <div className="flex flex-col md:flex-row justify-center items-stretch gap-x-4 gap-y-4 overflow-x-auto pb-2">
                  {selectedDisplayImages.map((imageId, imgIndex) => {
                    const imgInfo = getDisplayedImageInfo(imageId);
                    if (!imgInfo) return null;
                    const typeOptions = ['--- Select ---', '左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照'];
                    const qualityOptions = ['--- Select ---', '图像质量高', '图像质量可用', '图像质量差'];
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
                            value={imgInfo.quality}
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
                            checked={manualDiagnosis.left_eye[disease] || false}
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
                          value={customDiseases.left_eye || ''}
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
                            checked={manualDiagnosis.right_eye[disease] || false}
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
                          value={customDiseases.right_eye || ''}
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

            {/* (Old external submit/discard section removed; buttons now inside unified correction section) */}
          </div>
        )}
      </main>

      {/* Disclaimer */}
      <footer className="mt-8 text-center text-gray-600 text-xs">
        *人工智能系统存在一定局限性，可能产生误差，相关检测结果仅供参考，不构成最终决策依据。
        <br />
        (*AI system has certain limitations, may produce errors, and related detection results are for reference only, not as final decision basis.)
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
