# Driver Verification System

## Project Overview
This project is a FastAPI-based backend system designed for automated driver verification. It uses a robust computer vision pipeline to ensure drivers comply with safety and identity requirements. The system processes an uploaded image of a driver and their driver ID, verifies their identity against a face embedding database, and checks for the presence of a correct helmet, company logo, and whether the image might be a spoof (e.g., a fake screen photo). It also handles logging results and penalizing drivers who fail verification.

## System Workflow
1. **Receive Request**: The `/verify-driver` endpoint receives an image and a driver ID (`drv_id`).
2. **Person Detection**: A YOLO-based cropper detects and crops the upper body of the person in the image.
3. **Region Cropping**: Specialized croppers isolate the head, logo area, and the image border/background.
4. **Classification & Detection**:
   - Analyzes the head region to verify the presence of an approved helmet.
   - Analyzes the logo region to ensure the correct company logo is visible.
   - Analyzes the border region to detect whether the image was captured from a physical screen (anti-spoofing).
5. **Face Verification**: Extracts the face embedding using InsightFace and calculates the cosine similarity against the canonical embedding stored in a MySQL database.
6. **Decision Making**: The `DecisionModule` evaluates all predictions. The request is only deemed valid if:
   - A person is detected.
   - The helmet is valid (`helmet_1` or `helmet_2`).
   - The logo is detected (`logo`).
   - The image is not a fake screen (`true`).
   - The face matches the database (similarity $\ge$ 0.5).
7. **Result Handling**: The system logs the result, saves the processed image to the `captures` directory under `valid` or `invalid` subfolders, and updates the database. If a mismatch or violation occurs, the driver's violation count increments. After 3 violations, their account is deactivated.

## Main Modules in the Backend
- **`main.py`**: The FastAPI application entry point containing the `/verify-driver` endpoint.
- **`pipeline.py`** (`VisionPipeline`): Orchestrates the entire computer vision process, chaining together croppers, classifiers, and the face extraction module.
- **`decision_module.py`**: Contains the logic (`DecisionModule`) to determine if a pipeline run is valid and the handler (`ResultHandler`) to save captures/logs and update the database.
- **Croppers** (`cropper/`):
  - `person_cropper.py`: YOLOv8 based upper-body detector.
  - `head_cropper.py` / `logo_cropper.py`: MediaPipe based region of interest aligners.
  - `boder_cropper.py`: Extracts the borders of the image for spoof detection.
- **Classifiers**:
  - `helmet_classifier.py`: Validates the driver's helmet type.
  - `logo_classifier.py`: Validates the driver's uniform logo.
  - `fake_image_detector.py`: Detects spoofed images.
- **Embeddings** (`get_embed/`):
  - `get_face_embedding.py`: InsightFace pipeline for face feature extraction.
  - `get_embedding_mysql.py`: Manages the MySQL connection and retrieval of embeddings.

## Model Usage
The system integrates several optimized machine learning models:
- **Person Detection**: Ultralytics YOLO (`yolov8`)
- **Region isolation**: MediaPipe is utilized for semantic keypoint processing (head and logo cropping).
- **Helmet Classifier**: A fine-tuned ResNet18 (PyTorch) categorizing into `helmet_1`, `helmet_2`, or `wrong_helmet`.
- **Logo Classifier**: A custom PyTorch vision model categorizing into `logo` or `no_logo`.
- **Fake Image Detector**: A fine-tuned ResNet50 (PyTorch) preventing screen-spoofing attacks.
- **Face Recognition**: InsightFace (`FaceAnalysis`) generates embeddings.

> Note: All PyTorch models expect their weights to be located in the `models/` directory (e.g., `helmet_classification.pt`, `logo_classifier.pt`, `fake_screen_detector_resnet50.pt`).

## Database Usage
The project connects to a local MySQL instance (default: `localhost:3307`, database: `quanly_taixe`).
- **`embeddings`**: Stores the reference feature vectors (`embedding_vector`) mapped to each driver (`drv_id`).
- **`driver_images`**: Logs verification attempts, storing the image name, capture date, log file name, and whether the attempt was valid (`log_valid`).
- **`drive`**: Tracks the overall status of the drivers. Includes `num_violation` and `status_account`. If a driver fails the validation check making their `num_violation > 3`, their account is automatically marked as `deactive`.

## How to Run the Project Locally
Dowload models in this link bellow: https://drive.google.com/file/d/16sLCOpv3y7v31T1YPaTh8JLDpxdasbFv/view?usp=drive_link. Then creat a folder in .../backend/models, unzip all models in folder .../backend/models.
1. **Clone the repository and enter the backend directory**:
   ```bash
   cd Main/backend
   ```

2. **Set up a Python Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have generated or have the `requirements.txt` file setup for the environment.
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Models**:
   Ensure that all `.pt` model weights are placed inside the `Main/backend/models/` folder.

5. **Setup the Database**:
   - Install MySQL and create a database named `quanly_taixe`.
   - Update the connection parameters (port, user, password) inside `get_embed/get_embedding_mysql.py` and `decision_module.py` if they differ from the defaults (`port=3307`, `user=root`, `password=kunmuradstsv12`).
   - Define the `drive`, `driver_images`, and `embeddings` tables.

6. **Start the FastAPI Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Test the API**:
   You can submit a POST request to `http://localhost:8000/verify-driver` with `form-data` containing:
   - `drv_id`: (string) The driver's ID
   - `file`: (image) The live image captured of the driver
