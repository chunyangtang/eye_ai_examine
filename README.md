# Eye AI Examine

Obtain newly captured eye images, analyze them using deep learning model, and display the results in an web application.

## Usage

### Use Automation Script (Recommended)

You can use the provided `start_servers.sh` script to automatically start both the backend and frontend servers. This script will handle port selection and other setup tasks for you.

```bash
./start_servers.sh
```

### Manual - Backend

Run the backend server using:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

keep the terminal open for FastAPI to run on `http://127.0.0.1:8000`.

### Manual - Frontend

```bash
cd frontend/eye-frontend
npm install
npm start
```

Your browser should automatically open to `http://localhost:3000`.

For other computers in the local network, open `http://<your-ip-address>:3000`.

Use `http://<your-ip-address>:3000/?ris_exam_id=<exam_id>` to access a specific exam.

### LLM Configuration

The backend supports both the previous remote LLM API flow and the newer local Ollama deployment. Configuration lives in the `.env` file at the project root.

- `LLM_PROVIDER` – set to `ollama` for local models or `api` for any OpenAI-compatible HTTPS endpoint.
- `LLM_API_BASE` – the base URL (e.g. `http://localhost:11434` for Ollama or `https://api.openai.com` for OpenAI). Use `LLM_CHAT_ENDPOINT` if the chat path differs from the default (`/api/chat` for Ollama, `/v1/chat/completions` otherwise).
- `LLM_MODEL` – Ollama model tag (e.g. `DeepSeek-3.1:latest`) or remote model id (e.g. `gpt-4o-mini`).
- `LLM_API_KEY` – required only when `LLM_PROVIDER=api`.
- `LLM_TEMPERATURE` – optional; set only if the remote provider allows overriding temperature (otherwise the backend omits the field and the provider falls back to its default).

Restart the backend after changing these settings so FastAPI can pick up the new environment variables.


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
