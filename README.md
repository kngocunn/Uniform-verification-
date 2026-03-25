# Driver Verification System

## Project Overview  
This project is a FastAPI-based backend system for automated driver verification.  

It leverages a computer vision pipeline to ensure drivers meet safety and identity requirements. The system processes an uploaded image along with a driver ID (drv_id), verifies identity using a face embedding database, and checks:  

- Helmet compliance  
- Company logo presence  
- Anti-spoofing (detecting screen-based fake images)  

The system also logs results and penalizes drivers who fail verification.  

---

## System Workflow  

1. Receive Request  
The /verify-driver endpoint receives an image and driver ID.  

2. Person Detection  
A YOLO-based model detects and crops the upper body.  

3. Region Cropping  
- Head region  
- Logo region  
- Image border (for spoof detection)  

4. Classification & Detection  
- Helmet classification  
- Logo detection  
- Fake image detection  

5. Face Verification  
Extract face embeddings using InsightFace and compare with database (cosine similarity).  

6. Decision Making  
The request is valid only if:  
- Person detected  
- Helmet valid (helmet_1 or helmet_2)  
- Logo detected  
- Not a fake image  
- Face similarity ≥ 0.5  

7. Result Handling  
- Save image to captures/valid or captures/invalid  
- Log results into database  
- Increase violation count if failed  
- Deactivate account after 3 violations  

---

## Backend Structure  

backend/  
  main.py  
  pipeline.py  
  decision_module.py  
  cropper/  
    person_cropper.py  
    head_cropper.py  
    logo_cropper.py  
    boder_cropper.py  
  get_embed/  
    get_face_embedding.py  
    get_embedding_mysql.py  
  helmet_classifier.py  
  logo_classifier.py  
  fake_image_detector.py  

---

## Model Usage  

- Person Detection: YOLOv8 (Ultralytics)  
- Region Processing: MediaPipe  
- Helmet Classifier: ResNet18 (PyTorch)  
- Logo Classifier: Custom CNN (PyTorch)  
- Fake Image Detector: ResNet50 (PyTorch)  
- Face Recognition: InsightFace  

All model weights must be placed in:  
backend/models/  

---

## Database  

MySQL (default: localhost:3307, database: quanly_taixe)  

Tables:  
- embeddings  
- driver_images  
- drive  

---

## How to Run  

1. Download models  
Download from:  
https://drive.google.com/file/d/16sLCOpv3y7v31T1YPaTh8JLDpxdasbFv/view  

Unzip into:  
backend/models/  

---

2. Setup environment  

cd Main/backend  
python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

---

3. Setup database  

- Create database: quanly_taixe  
- Update credentials in:  
  get_embedding_mysql.py  
  decision_module.py  

---

4. Run server  

uvicorn main:app --host 0.0.0.0 --port 8000 --reload  

---

5. Test API  

POST:  
http://localhost:8000/verify-driver  

Form-data:  
- drv_id (string)  
- file (image)  

---

## Demo  
(Add screenshots or demo here)  

---

## Future Improvements  
- Improve model accuracy  
- Add real-time processing  
- Deploy to cloud  
