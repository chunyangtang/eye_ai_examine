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


function App() {
  const [patientData, setPatientData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDisplayImages, setSelectedDisplayImages] = useState([]); // Stores image IDs for the 4 display slots
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitMessage, setSubmitMessage] = useState('');

  const patientIdPrefix = "病人ID (Patient ID): ";

  const diseaseNames = {
    糖网: 'Diabetic Retinopathy',
    青光眼: 'Glaucoma',
    AMD: 'AMD',
    病理性近视: 'Pathological Myopia',
    RVO: 'RVO',
    RAO: 'RAO',
    视网膜脱离: 'Retinal Detachment',
    其它黄斑病变: 'Other Macular Diseases',
    其它眼病变: 'Other Eye Diseases',
    正常: 'Normal',
  };

  const fetchPatientData = useCallback(async (endpoint) => {
    setLoading(true);
    setError(null);
    setPatientData(null);
    setSelectedDisplayImages([]); // Clear previous selection
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/patients/${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
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

    } catch (e) {
      console.error("Failed to fetch patient data:", e);
      setError(`Failed to load patient data: ${e.message}. Is the Python server running?`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPatientData('current');
  }, [fetchPatientData]);

  // Handler for changes to Image Type or Image Quality dropdowns
  const handleImageDetailChange = (imageId, field, value) => {
    setPatientData(prevData => {
      if (!prevData) return null;
      const updatedEyeImages = prevData.eye_images.map(img =>
        img.id === imageId ? { ...img, [field]: value } : img
      );
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
      return { ...prevData, diagnosis_results: newDiagnosisResults };
    });
  };

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
          eye_images: patientData.eye_images, // UPDATED: Submit image type/quality changes
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setSubmitMessage(result.status);
    } catch (e) {
      console.error("Failed to submit diagnosis:", e);
      setSubmitMessage(`Error submitting: ${e.message}`);
    } finally {
      setIsSubmitting(false);
      // Optionally re-fetch current patient data to confirm save
      // fetchPatientData('current');
    }
  };

  const handleReselectImages = (newSelectedIds) => {
    setSelectedDisplayImages(newSelectedIds);
  };

  const getDisplayedImageInfo = (imageId) => {
    // Look up the image info from the current patientData.eye_images
    // This allows the dropdowns to reflect local state changes
    return patientData?.eye_images.find(img => img.id === imageId);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 p-6 font-inter text-gray-800 antialiased">
      {/* Header */}
      <header className="bg-white rounded-xl shadow-lg p-4 mb-8 flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-xs mr-4 shadow-inner">
            Tsinghua <br /> Logo
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-gray-900 tracking-tight">
            Tsinghua BBNC AI眼科辅助诊断系统
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
            <div className="flex flex-col md:flex-row justify-between items-center mb-8 gap-4">
              <h2 className="text-xl md:text-2xl font-semibold text-gray-700">
                {patientIdPrefix}<span className="text-blue-700 font-bold">{patientData.patient_id}</span>
              </h2>
              {/* Reselect Image button is now here, in the patient info row */}
              <button
                onClick={() => setIsModalOpen(true)}
                className="px-5 py-2 bg-green-600 text-white rounded-lg shadow-md
                hover:bg-green-700 transition-all duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                重新选择图片 (Reselect Image)
              </button>
            </div>

            {/* Image Display Section */}
            <div className="relative flex items-center justify-center mb-10 bg-gray-50 p-6 rounded-xl shadow-inner">
              {/* Left arrow for previous patient */}
              <IconButton onClick={() => fetchPatientData('previous')} className="absolute left-3 z-10">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </IconButton>

              <div className="flex flex-wrap justify-center gap-6 mx-12"> {/* Increased horizontal margin for arrows */}
                {selectedDisplayImages.map((imageId, index) => {
                  const imgInfo = getDisplayedImageInfo(imageId);
                  return (
                    <div key={index} className="flex-shrink-0 w-48 h-56 bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 transform hover:scale-105 transition-transform duration-200 group">
                      {imgInfo ? (
                        <>
                          <img
                            src={`data:image/png;base64,${imgInfo.base64_data}`}
                            alt={`Image ${index + 1}`}
                            className="w-full h-40 object-cover border-b border-gray-200 group-hover:opacity-80 transition-opacity"
                          />
                          <div className="p-3 text-sm text-center">
                            <p className="font-semibold text-gray-900 group-hover:text-blue-700 transition-colors">IMAGE {index + 1}</p>
                            <p className="text-gray-600">{imgInfo.type}</p>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-56 bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
                          No Image Selected
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Right arrow for next patient */}
              <IconButton onClick={() => fetchPatientData('next')} className="absolute right-3 z-10">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </IconButton>
            </div>

            {/* Image Details (Type and Quality) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6 mb-10">
              {['影像类型 (Image Type)', '影像质量 (Image Quality)'].map((label, colIndex) => (
                <div key={label} className="col-span-1 bg-white p-5 rounded-lg shadow-sm border border-gray-100">
                  <h3 className="text-lg font-semibold mb-4 text-gray-700">{label}</h3>
                  {selectedDisplayImages.map((imageId, imgIndex) => {
                    const imgInfo = getDisplayedImageInfo(imageId);
                    if (!imgInfo) return null;

                    const options = colIndex === 0
                      ? ['左眼CFP', '右眼CFP', '左眼外眼照', '右眼外眼照', 'OCT', 'FFA']
                      : ['Good', 'Usable', 'Bad'];

                    const currentValue = colIndex === 0 ? imgInfo.type : imgInfo.quality;
                    const fieldToUpdate = colIndex === 0 ? 'type' : 'quality';

                    return (
                      <div key={imgIndex} className="flex items-center mb-3">
                        <span className="w-20 text-right pr-4 font-medium text-gray-600 text-sm">Image {imgIndex + 1}:</span>
                        <select
                          value={currentValue}
                          onChange={(e) => handleImageDetailChange(imgInfo.id, fieldToUpdate, e.target.value)}
                          className="flex-grow p-2 border border-gray-300 rounded-md bg-white text-gray-700 text-sm
                            focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-150 shadow-sm"
                        >
                          {options.map(option => (
                            <option key={option} value={option}>{option}</option>
                          ))}
                        </select>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>


            {/* Diagnosis Results Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8 mb-10">
              {['left_eye', 'right_eye'].map(eye => (
                <div key={eye} className="border border-gray-200 rounded-xl p-6 shadow-md bg-gray-50">
                  <h3 className="text-xl font-semibold mb-5 text-center text-gray-800">
                    {eye === 'left_eye' ? '左眼疾病诊断 (Left Eye Disease Diagnosis)' : '右眼疾病诊断 (Right Eye Disease Diagnosis)'}
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    {Object.entries(patientData.diagnosis_results[eye]).map(([diseaseKey, isDetected]) => (
                      <div key={diseaseKey} className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm border border-gray-100">
                        <span className="text-base text-gray-700 font-medium">{diseaseNames[diseaseKey]}</span>
                        <label className="flex items-center cursor-pointer">
                          <div className="relative">
                            <input
                              type="checkbox"
                              className="sr-only peer"
                              checked={isDetected}
                              onChange={() => handleDiagnosisToggle(eye, diseaseKey)}
                            />
                            <div className="block w-11 h-6 bg-gray-300 rounded-full peer-checked:bg-blue-600 transition-colors duration-300"></div>
                            <div className="absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform duration-300 peer-checked:translate-x-5 shadow-sm"></div>
                          </div>
                        </label>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Submit/Discard Buttons */}
            <div className="bg-gray-100 p-6 rounded-xl flex flex-col items-center justify-center space-y-5 md:flex-row md:space-y-0 md:space-x-6 shadow-inner">
              <p className="text-md font-medium text-gray-700">
                检测结果有误？可在页面中修正后提交 (Detection results are incorrect? You can modify and submit on this page)
              </p>
              <div className="flex space-x-4">
                <button
                  onClick={() => { /* Implement discard logic, e.g., re-fetch original data */ }}
                  className="px-7 py-3 bg-red-500 text-white rounded-lg shadow-md hover:bg-red-600 transition-colors duration-200 text-lg font-semibold"
                >
                  放弃 (Discard)
                </button>
                <button
                  onClick={handleSubmitDiagnosis}
                  disabled={isSubmitting}
                  className={`px-7 py-3 rounded-lg shadow-md transition-colors duration-200 text-lg font-semibold
                  ${isSubmitting ? 'bg-green-300 cursor-not-allowed' : 'bg-green-600 text-white hover:bg-green-700'}`}
                >
                  {isSubmitting ? '提交中...' : '提交 (Submit)'}
                </button>
              </div>
            </div>
            {submitMessage && (
              <p className="mt-6 text-center text-md font-medium text-green-700">{submitMessage}</p>
            )}
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
    </div>
  );
}

export default App;
