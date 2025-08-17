from flask import Flask, render_template, request, redirect
import os
import base64
from PIL import Image
import io
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Roboflow API - Banana classification model
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="jCTsCrnUhceaGPAS2P7x"
)
MODEL_ID = "banana-ripeness-classification-46ikl/3"  # ResNet50 classification model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Nếu ảnh từ webcam
        if 'cam_image' in request.form and request.form['cam_image']:
            data_url = request.form['cam_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(binary_data)).convert("RGB")
            filename = "webcam.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
        else:
            # Nếu upload từ máy
            file = request.files["image"]
            if file.filename == "":
                return redirect(request.url)
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Gọi Roboflow API (classification)
        try:
            result = CLIENT.infer(filepath, model_id=MODEL_ID)
        except Exception as e:
            return f"Lỗi gọi API Roboflow: {e}"

        # Lấy kết quả dự đoán
        predictions = result.get("predictions", [])
        if predictions:
            best_pred = max(predictions, key=lambda p: p["confidence"])
            predicted_class = best_pred["class"]
            confidence = best_pred["confidence"]
             # ✅ Nếu là unknown → hiển thị rõ ràng là “Không xác định được”
            if predicted_class.lower() == "unknown":
                predicted_class = "Undetermined"
                confidence = 0
        else:
            predicted_class = "Không xác định"
            confidence = 0

        return render_template(
            "result.html",
            image_filename=filename,
            predicted_class=predicted_class,
            confidence=confidence
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
