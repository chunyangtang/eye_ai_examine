# trigger_data_update.py
import requests
from patientdataio import create_single_dummy_patient_data

# URL of your FastAPI backend endpoint to add new patient data
BACKEND_URL = "http://127.0.0.1:8000/api/add_new_patient"


def send_new_patient_data():
    """Generates a new patient's data and sends it to the backend."""
    patient_data = create_single_dummy_patient_data()
    print(f"Generating new patient data for ID: {patient_data.patient_id}")
    print(f"Attempting to send data to backend at {BACKEND_URL}...")
    try:
        response = requests.post(BACKEND_URL, json=patient_data.dict()) # Send Pydantic model as dict
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        print(f"Backend response: {response.json().get('status', 'Success')}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to the backend. Is FastAPI running at {BACKEND_URL}?")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred during the request: {e}")

if __name__ == "__main__":
    print("This script will send new patient data to the FastAPI backend.")
    print("Press Enter to send a new patient's data (or type 'exit' to quit).")
    while True:
        user_input = input("Action (press Enter or type 'exit'): ")
        if user_input.lower() == 'exit':
            print("Exiting trigger script.")
            break
        else:
            send_new_patient_data()
            print("\nWaiting for next patient data input...")
