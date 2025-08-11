# Eye AI Examine

Obtain newly captured eye images, analyze them using deep learning model, and display the results in an web application.

## Usage
### Backend

Run the backend server using:

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

keep the terminal open for FastAPI to run on `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend/eye-frontend
npm install
npm start
```

Your browser should automatically open to `http://localhost:3000`.

