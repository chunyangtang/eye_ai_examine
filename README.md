# Eye AI Examine

Obtain newly captured eye images, analyze them using deep learning model, and display the results in an web application.

## Usage
### Backend

Run the backend server using:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

keep the terminal open for FastAPI to run on `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend/eye-frontend
npm install
npm start
```

Your browser should automatically open to `http://localhost:3000`.

For other computers in the local network, open `http://<your-ip-address>:3000`.

Use `http://<your-ip-address>:3000/?ris_exam_id=<exam_id>` to access a specific exam.


## Description

### Backend
The backend is built with FastAPI and serves as the API for the frontend. It handles patient data, image processing, and model inference.

**Files**
- `main.py`: The entry point for the FastAPI application.
- `patientdataio.py`: Handles reading and writing patient data.
- `datatype.py`: Defines the data types used in the application.

As for now (2025.8.14), the backend is defaulted to use data from `{project_root}/data` for existing exams.

### Frontend
The frontend is built with React and provides the user interface for the application. It communicates with the backend API to fetch and display data.

**Files**
- `src/App.js`: The main application component.
