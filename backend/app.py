
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from werkzeug.utils import secure_filename

from preprocessing import preprocess_image
from database import create_database, save_xray_record, get_all_records
from predict import predict_xray, InvalidImageError

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Automatically create the database when the backend starts.
create_database()


@app.route("/")
def home():
    return "AI Medical Imaging Backend is running!"


@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(image.filename)

    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)

    try:
        prediction, confidence = predict_xray(file_path)

        processed_image = preprocess_image(file_path)

        save_xray_record(filename, prediction, confidence)

        return jsonify({
            "message": "X-ray processed successfully.",
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "shape": list(processed_image.shape)
        }), 200

    except InvalidImageError as error:

        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"error": str(error)}), 400

    except Exception as error:
        return jsonify({
            "error": "Image processing failed: " + str(error)
        }), 500


@app.route("/history", methods=["GET"])
def scan_history():

    records = get_all_records()

    history = []

    for record in records:
        history.append({
            "id": record[0],
            "filename": record[1],
            "date": record[2],
            "prediction": record[3],
            "confidence": record[4]
        })

    return jsonify(history)


@app.route("/stats", methods=["GET"])
def dashboard_stats():

    records = get_all_records()

    total_scans = len(records)

    normal_count = sum(
        1 for record in records if record[3] == "normal"
    )

    pneumonia_count = sum(
        1 for record in records if record[3] == "pneumonia"
    )

    return jsonify({
        "total_scans": total_scans,
        "normal_count": normal_count,
        "pneumonia_count": pneumonia_count
    })


if __name__ == "__main__":
    app.run(debug=True)
