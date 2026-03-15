from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2

from pipeline import VisionPipeline
from decision_module import DecisionModule, ResultHandler

app = FastAPI()

# ===== ADD CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODULES =====
pipeline = VisionPipeline()
decision_module = DecisionModule()
result_handler = ResultHandler()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/verify-driver")
async def verify_driver(
    drv_id: str = Form(...),
    file: UploadFile = File(...)
):

    image_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = pipeline.run(image_path, drv_id)

    if "error" in result:
        return {"status": "error", "message": result["error"]}

    decision = decision_module.evaluate(result)

    image = cv2.imread(image_path)

    saved_path = result_handler.process(image, drv_id, decision, result)

    return {
        "pipeline_result": result,
        "decision": decision,
        "saved_image": saved_path
    }