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

  const patientIdPrefix = "病人索引 (Patient Index): ";

  const diseaseNames = {
    青光眼: 'Glaucoma',
    糖网: 'Diabetic Retinopathy',
    AMD: 'AMD',
    病理性近视: 'Pathological Myopia',
    RVO: 'RVO',
    RAO: 'RAO',
    视网膜脱离: 'Retinal Detachment',
    其它视网膜病: 'Other Retinal Diseases',
    其它黄斑病变: 'Other Macular Diseases',
    白内障: 'Cataract',
    正常: 'Normal',
  };

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
  const isDataEqual = (a, b) => {
    return JSON.stringify(a) === JSON.stringify(b);
  };

  const fetchPatientData = useCallback(async (endpoint, { skipIfUnsaved = false } = {}) => {
    if (skipIfUnsaved && hasUnsavedChanges) return;
    let willUpdate = false;
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/patients/${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      // Only update if data is different and no unsaved changes
      if (!hasUnsavedChanges && (!patientData || !isDataEqual(result, patientData))) {
        willUpdate = true;
        setLoading(true);
        setPatientData(result);
        // Initialize selectedDisplayImages
        const uniqueImageTypes = {};
        result.eye_images.forEach(img => {
          if (!uniqueImageTypes[img.type]) {
            uniqueImageTypes[img.type] = img.id;
          }
        });
        let initialSelection = Object.values(uniqueImageTypes);
        // Ensure there are exactly 4 images for display
        while (initialSelection.length < 4 && result.eye_images.length > 0) {
          // If less than 4 unique types, auto display the same kind ones if available
          const availableImages = result.eye_images.filter(img => !initialSelection.includes(img.id));
          if (availableImages.length > 0) {
            initialSelection.push(availableImages[0].id);
          } else {
            // If no new images to add, break to prevent infinite loop for patients with <4 images
            break;
          }
        }
        setSelectedDisplayImages(initialSelection.slice(0, 4));
      }
    } catch (e) {
      console.error("Failed to fetch patient data:", e);
      setError(`Failed to load patient data: ${e.message}. Is the Python server running?`);
    } finally {
      if (willUpdate) setLoading(false);
    }
  }, [hasUnsavedChanges, patientData]);

  // Polling effect
  useEffect(() => {
    fetchPatientData('current'); // Initial fetch
    const intervalId = setInterval(() => {
      fetchPatientData('current', { skipIfUnsaved: true });
    }, 5000); // Poll every 5 seconds
    return () => clearInterval(intervalId);
  }, [fetchPatientData]);

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

  const handleDiagnosisToggle = (eye, disease) => {
    setPatientData(prevData => {
      const newDiagnosisResults = { ...prevData.diagnosis_results };
      newDiagnosisResults[eye] = {
        ...newDiagnosisResults[eye],
        [disease]: !newDiagnosisResults[eye][disease]
      };
      setHasUnsavedChanges(true);
      return { ...prevData, diagnosis_results: newDiagnosisResults };
    });
  };

  const handleReselectImages = (newSelectedIds) => {
    setSelectedDisplayImages(newSelectedIds);
    setHasUnsavedChanges(true);
  };

  const handleDiscardChanges = () => {
    setSubmitMessage(''); // Clear any previous submit messages
    setHasUnsavedChanges(false);
    fetchPatientData('current'); // Reload original data for the current patient
  };

  // Submit both diagnosis and image info in one request
  const handleSubmitDiagnosis = async () => {
    if (!patientData) return;
    setIsSubmitting(true);
    setSubmitMessage('');
    try {
      const response = await fetch('http://127.0.0.1:8000/api/submit_diagnosis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientData.patient_id,
          diagnosis_results: patientData.diagnosis_results,
          image_updates: patientData.eye_images.map(({id, type, quality}) => ({id, type, quality}))
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setSubmitMessage(result.status);
      setHasUnsavedChanges(false);
      fetchPatientData('current');
    } catch (e) {
      console.error('Failed to submit diagnosis:', e);
      setSubmitMessage(`Error submitting: ${e.message}`);
    } finally {
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
          </div>
        )}

        {patientData && (
          <div>
            {/* Patient ID and Reselect Image Button */}
            <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-2"> {/* Shrinked gap */}
              <span className="text-sm text-gray-500"> {/* Changed to span for less prominence */}
                {patientIdPrefix}<span className="text-blue-700 font-bold">{patientData.patient_id}</span>
              </span>
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
              {/* Left arrow for previous patient */}
              <IconButton onClick={() => fetchPatientData('previous')} className="absolute left-3 z-10">
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

              {/* Right arrow for next patient */}
              <IconButton onClick={() => fetchPatientData('next')} className="absolute right-3 z-10">
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
                      <th className="w-28 p-2 text-left text-gray-600 font-medium">疾病 (Disease)</th>
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
                          <td className="p-2 align-middle text-gray-700 whitespace-nowrap font-medium">{diseaseNames[dk]}</td>
                          {/* Left eye bar */}
                          <td className="p-2">
                            <div className="relative h-5 rounded-full bg-gradient-to-r from-green-300 via-yellow-300 to-red-400 overflow-hidden shadow-inner">
                              {/* Fill overlay */}
                              <div className="absolute top-0 left-0 h-full bg-green-600/20" style={{ width: `${leftWidth}%` }} />
                              {/* Threshold marker at 50% */}
                              <div className="absolute top-0 left-1/2 w-0.5 h-full bg-gray-600/70" />
                              {/* Probability marker (line + diamond) */}
                              <div className="absolute top-0 h-full flex items-center" style={{ left: `${leftWidth}%`, transform: 'translateX(-50%)' }}
                                   title={`P:${formatProb(leftProb)} T:${formatProb(t)}`}
                                   aria-label={`Left eye ${diseaseNames[dk]} probability ${formatProb(leftProb)}, threshold ${formatProb(t)}`}>
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
                                   title={`P:${formatProb(rightProb)} T:${formatProb(t)}`}
                                   aria-label={`Right eye ${diseaseNames[dk]} probability ${formatProb(rightProb)}, threshold ${formatProb(t)}`}>
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
                    const qualityOptions = ['--- Select ---', 'Good', 'Usable', 'Bad'];
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
              {/* Diagnosis Grid */}
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-3 tracking-wide">疾病诊断 (Disease Diagnosis)</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {['left_eye', 'right_eye'].map(eye => (
                    <div key={eye} className="border border-gray-100 rounded-lg p-4 bg-gray-50">
                      <h5 className="text-sm font-semibold mb-4 text-center text-gray-800">
                        {eye === 'left_eye' ? '左眼 (Left Eye)' : '右眼 (Right Eye)'}
                      </h5>
                      <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                        {Object.entries(patientData.diagnosis_results[eye]).map(([diseaseKey, isDetected]) => (
                          <button
                            key={diseaseKey}
                            onClick={() => handleDiagnosisToggle(eye, diseaseKey)}
                            className={`px-2 py-2 rounded-md shadow-sm border text-center text-[11px] md:text-xs font-medium tracking-tight
                              ${isDetected ? 'bg-blue-600 text-white border-blue-700 hover:bg-blue-700' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-100'}
                              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 transition-colors`}
                          >
                            {diseaseNames[diseaseKey]}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
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
