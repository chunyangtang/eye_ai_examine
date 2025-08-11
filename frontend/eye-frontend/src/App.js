// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;


import React, { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [messageToSend, setMessageToSend] = useState('');
  const [backendResponse, setBackendResponse] = useState('');
  const [storedMessages, setStoredMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);

  // Effect to fetch initial data (message and image) from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setData(result);
      } catch (e) {
        console.error("Failed to fetch initial data:", e);
        setError("Failed to load initial data from backend. Is the Python server running?");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Effect to fetch stored messages from backend when component mounts or a message is sent
  useEffect(() => {
    const fetchStoredMessages = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/messages');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setStoredMessages(result.messages);
      } catch (e) {
        console.error("Failed to fetch stored messages:", e);
        // Optionally set an error specifically for this part if needed
      }
    };

    fetchStoredMessages();
  }, [backendResponse]); // Re-fetch when backendResponse changes (i.e., after sending a message)

  // Handler for sending message to backend
  const handleSendMessage = async () => {
    if (!messageToSend.trim()) {
      alert('Please enter a message to send.'); // Using alert for a tiny demo, consider custom modal for real app
      return;
    }

    setIsSending(true);
    setBackendResponse('');
    try {
      const response = await fetch('http://127.0.0.1:8000/api/send_message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: messageToSend }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setBackendResponse(result.status);
      setMessageToSend(''); // Clear the input field
    } catch (e) {
      console.error("Failed to send message:", e);
      setBackendResponse(`Error: ${e.message}`);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-lg shadow-xl max-w-lg w-full text-center">
        <h1 className="text-3xl font-extrabold text-gray-900 mb-6">
          Frontend-Backend Interaction Demo
        </h1>

        {loading && (
          <p className="text-blue-500 text-lg">Loading initial data from backend...</p>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline ml-2">{error}</span>
          </div>
        )}

        {/* Display initial data (image and text) from backend */}
        {data && (
          <div className="mb-8 p-4 border border-gray-200 rounded-lg bg-gray-50">
            <h2 className="text-2xl font-semibold text-gray-800 mb-3">Initial Data (Backend to Frontend)</h2>
            <p className="text-gray-700 text-lg mb-4">
              {data.message}
            </p>
            {data.image_base64 && (
              <img
                src={`data:image/png;base64,${data.image_base64}`}
                alt="Image from Backend"
                className="mx-auto rounded-lg shadow-md border-2 border-green-400"
              />
            )}
            {!data.image_base64 && (
              <p className="text-yellow-600 mt-4">No image data received from backend.</p>
            )}
          </div>
        )}

        {/* Frontend to Backend Interaction */}
        <div className="mb-8 p-4 border border-blue-200 rounded-lg bg-blue-50">
          <h2 className="text-2xl font-semibold text-gray-800 mb-3">Send Message (Frontend to Backend)</h2>
          <input
            type="text"
            className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
            placeholder="Type your message here..."
            value={messageToSend}
            onChange={(e) => setMessageToSend(e.target.value)}
          />
          <button
            onClick={handleSendMessage}
            disabled={isSending}
            className={`w-full px-4 py-2 rounded-md font-semibold text-white transition-colors duration-200 ${
              isSending ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isSending ? 'Sending...' : 'Send Message to Backend'}
          </button>
          {backendResponse && (
            <p className="mt-4 text-green-700 font-medium">Backend Says: {backendResponse}</p>
          )}
        </div>

        {/* Display preserved messages from backend */}
        <div className="p-4 border border-purple-200 rounded-lg bg-purple-50">
          <h2 className="text-2xl font-semibold text-gray-800 mb-3">Messages Stored in Backend (Simulated Preservation)</h2>
          {storedMessages.length > 0 ? (
            <ul className="list-disc list-inside text-left text-gray-700">
              {storedMessages.map((msg, index) => (
                <li key={index} className="mb-1">{msg}</li>
              ))}
            </ul>
          ) : (
            <p className="text-gray-600">No messages stored yet. Send one above!</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
