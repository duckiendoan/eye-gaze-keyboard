# Eye-Gaze Keyboard

This repository provides a virtual keyboard controlled by eye gaze estimation or mouse input. Follow the instructions below to set up and run the project.

## Prerequisites

Ensure you have the following installed on your system:

- Python (3.7 or higher)
- pip (Python package installer)

## Setup Instructions

1. **Create a Python Virtual Environment**

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

2. **Install Required Packages**

   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Run the Keyboard with Eye Gaze Estimation

   Execute the following command:
   ```bash
   python src/main.py
   ```

   This mode uses eye gaze tracking to control the keyboard.

### 2. Run the Keyboard Demo with Mouse Control

   Execute the following command:
   ```bash
   python src/gaze_keyboard.py
   ```

   This mode allows you to use a mouse as an input to interact with the keyboard.

### Optional: Run Camera Calibration Using Checkerboard

   If you need to calibrate the camera, run the following command:
   ```bash
   python camera_data/main_camera_calibration.py
   ```

   This step is useful for improving accuracy when using eye gaze estimation.

## Additional Notes

- Make sure your environment supports the required hardware and software dependencies for eye gaze estimation.
- For any issues or questions, refer to the project documentation or open an issue in the repository.
