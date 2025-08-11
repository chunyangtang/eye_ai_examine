# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from base64 import b64encode
import io
from PIL import Image

app = FastAPI()

# Configure CORS to allow requests from your React frontend
# In a real application, replace "http://localhost:3000" with your frontend's actual domain
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for messages (simulates a database/file for this demo)
# In a real application, you would use a proper database or file system for persistence.
stored_messages = []

# Pydantic model for incoming message
class Message(BaseModel):
    message: str

# Function to generate a simple placeholder image and encode it to base64
def generate_placeholder_image_base64():
    """Generates a small green square image and returns its Base64 string."""
    width, height = 150, 150
    # Create a green image
    img = Image.new('RGB', (width, height), color = 'green')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode("utf-8")

# Store the generated base64 image string once to avoid re-generating on every request
STATIC_IMAGE_BASE64 = generate_placeholder_image_base64()

@app.get("/api/data")
async def get_initial_data():
    """
    Returns a simple message and a Base64 encoded image string for initial display.
    """
    print("Received request for /api/data")
    return {
        "message": "Hello from Python! This image and text came from the backend.",
        "image_base64": STATIC_IMAGE_BASE64
    }

@app.post("/api/send_message")
async def send_message(msg: Message):
    """
    Receives a message from the frontend, "preserves" it in memory,
    and sends a confirmation back.
    """
    print(f"Received message from frontend: {msg.message}")
    stored_messages.append(msg.message) # Simulate saving the message
    return {"status": f"Message received: '{msg.message}' and preserved!"}

@app.get("/api/messages")
async def get_messages():
    """
    Returns all messages currently "preserved" in memory.
    """
    print("Received request for /api/messages")
    return {"messages": stored_messages}

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application
    uvicorn.run(app, host="127.0.0.1", port=8000)
