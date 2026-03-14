import json
import os
import cv2
import mysql.connector
from datetime import datetime


class DecisionModule:
    
    def evaluate(self, result, drv_id=None):

        if not result.get("person_detected", False):
            return {"valid": False, "reason": "person_not_detected"}

        helmet = result.get("helmet", "").lower()
        if helmet not in ["helmet_1", "helmet_2"]:
            return {"valid": False, "reason": "invalid_helmet"}

        logo = result.get("logo", "").lower()
        if logo != "logo":
            return {"valid": False, "reason": "logo_invalid"}

        fake_screen = result.get("fake_screen", "").lower()
        if fake_screen != "true":
            return {"valid": False, "reason": "fake_screen_detected"}

        if not result.get("face_match", False):
            return {"valid": False, "reason": "face_not_match"}

        return {"valid": True}


class ResultHandler:

    def __init__(self, save_dir="captures"):

        self.save_dir = save_dir

        self.db = mysql.connector.connect(
            host="localhost",
            port=3307,
            user="root",
            password="kunmuradstsv12",
            database="quanly_taixe"
        )

    def process(self, image, drv_id, decision, result):

        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")

        driver_folder = os.path.join(self.save_dir, str(drv_id))

        image_folder = os.path.join(driver_folder, "image")

        # ===== chia log folder =====
        log_valid_folder = os.path.join(driver_folder, "log", "valid")
        log_invalid_folder = os.path.join(driver_folder, "log", "invalid")

        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(log_valid_folder, exist_ok=True)
        os.makedirs(log_invalid_folder, exist_ok=True)

        # ===== chọn folder theo decision =====
        if decision["valid"]:
            log_folder = log_valid_folder
        else:
            log_folder = log_invalid_folder

        image_name = f"{drv_id}_{date_str}.jpg"
        image_path = os.path.join(image_folder, image_name)

        log_name = f"{drv_id}_{date_str}.json"
        log_path = os.path.join(log_folder, log_name)

        # ===== SAVE IMAGE =====
        cv2.imwrite(image_path, image)

        # ===== SAVE LOG =====
        log_data = {
            "driver_id": drv_id,
            "time": str(now),
            "pipeline_result": result,
            "decision": decision
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

        cursor = self.db.cursor()

        cursor.execute(
            "INSERT IGNORE INTO drive (drv_id) VALUES (%s)",
            (drv_id,)
        )

        cursor.execute(
            """
            INSERT INTO driver_images (drv_id, image_name, capture_date, log_name, log_valid)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                drv_id,
                image_name,
                now,
                log_name,
                1 if decision["valid"] else 0
            )
        )

        if not decision["valid"]:

            query = """
            UPDATE drive
            SET num_violation = num_violation + 1,
                status_account = CASE
                    WHEN num_violation + 1 > 3 THEN 'deactive'
                    ELSE status_account
                END
            WHERE drv_id = %s
            """

            cursor.execute(query, (drv_id,))

        self.db.commit()
        cursor.close()

        return image_path