import os
import io
import base64
import tempfile
import numpy as np
import cv2
import rasterio
import tensorflow as tf
from flask import Flask, request, render_template_string
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "best_model.keras"
PERCENTILES_PATH = "band_percentiles.npz"

EXPECTED_CHANNELS = 12
THRESHOLD = 0.48

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(PERCENTILES_PATH):
    raise FileNotFoundError(f"Percentiles file not found: {PERCENTILES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

vals = np.load(PERCENTILES_PATH)
P_LOW = vals["p_low"].astype(np.float32)
P_HIGH = vals["p_high"].astype(np.float32)

if len(P_LOW) != EXPECTED_CHANNELS or len(P_HIGH) != EXPECTED_CHANNELS:
    raise ValueError(
        f"Expected {EXPECTED_CHANNELS} channels, got len(P_LOW)={len(P_LOW)}, len(P_HIGH)={len(P_HIGH)}"
    )

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Segmentation</title>
    <style>
        *{
            box-sizing:border-box;
        }

        body{
            margin:0;
            font-family:Inter, Arial, sans-serif;
            background:linear-gradient(135deg, #eef4ff 0%, #f7fbff 100%);
            color:#1f2937;
        }

        .page{
            max-width:1400px;
            margin:0 auto;
            padding:32px 20px 40px;
        }

        .hero{
            background:linear-gradient(135deg, #0f4c81 0%, #155e9a 100%);
            color:white;
            border-radius:24px;
            padding:28px 28px 24px;
            box-shadow:0 18px 50px rgba(15, 76, 129, 0.22);
            margin-bottom:22px;
        }

        .hero h1{
            margin:0 0 10px;
            font-size:32px;
            font-weight:800;
            letter-spacing:-0.5px;
        }

        .hero p{
            margin:0;
            font-size:15px;
            line-height:1.8;
            opacity:0.95;
        }

        .panel{
            background:#ffffff;
            border-radius:22px;
            padding:24px;
            box-shadow:0 14px 40px rgba(15, 23, 42, 0.08);
            border:1px solid rgba(226,232,240,0.8);
        }

        .upload-box{
            border:2px dashed #bfd7ee;
            background:#f8fbff;
            border-radius:20px;
            padding:24px;
            margin-bottom:16px;
            transition:0.2s ease;
        }

        .upload-box:hover{
            border-color:#7cb0da;
            background:#f4f9ff;
        }

        .upload-title{
            font-size:18px;
            font-weight:700;
            margin-bottom:8px;
            color:#0f4c81;
        }

        .upload-sub{
            color:#5b6677;
            font-size:14px;
            margin-bottom:18px;
        }

        .file-row{
            display:flex;
            flex-wrap:wrap;
            gap:14px;
            align-items:center;
        }

        input[type="file"]{
            flex:1 1 340px;
            background:white;
            border:1px solid #d7e3f0;
            border-radius:14px;
            padding:12px;
            color:#334155;
        }

        .btn{
            border:none;
            background:linear-gradient(135deg, #0f4c81 0%, #14649f 100%);
            color:white;
            padding:13px 22px;
            border-radius:14px;
            cursor:pointer;
            font-size:15px;
            font-weight:700;
            box-shadow:0 10px 24px rgba(15, 76, 129, 0.22);
            transition:transform 0.15s ease, box-shadow 0.15s ease;
        }

        .btn:hover{
            transform:translateY(-1px);
            box-shadow:0 14px 28px rgba(15, 76, 129, 0.28);
        }

        .error{
            margin-top:14px;
            background:#fff1f1;
            color:#b42318;
            border:1px solid #f5c2c7;
            padding:14px 16px;
            border-radius:14px;
            font-size:14px;
        }

        .meta-grid{
            display:grid;
            grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
            gap:14px;
            margin-top:20px;
            margin-bottom:22px;
        }

        .metric{
            background:#f8fbff;
            border:1px solid #e3edf7;
            border-radius:18px;
            padding:16px;
        }

        .metric .label{
            color:#64748b;
            font-size:13px;
            margin-bottom:8px;
        }

        .metric .value{
            color:#0f172a;
            font-size:18px;
            font-weight:800;
            word-break:break-word;
        }

        .section-title{
            margin:6px 0 16px;
            font-size:20px;
            font-weight:800;
            color:#0f172a;
        }

        .cards{
            display:grid;
            grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
            gap:18px;
        }

        .card{
            background:#ffffff;
            border:1px solid #e8eef5;
            border-radius:22px;
            padding:16px;
            box-shadow:0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .card-title{
            margin:0 0 12px;
            font-size:18px;
            font-weight:800;
            color:#0f4c81;
        }

        .image-wrap{
            background:#f8fafc;
            border-radius:16px;
            padding:10px;
            border:1px solid #edf2f7;
            display:flex;
            justify-content:center;
            align-items:center;
            min-height:220px;
        }

        .image-wrap img{
            max-width:100%;
            width:100%;
            height:auto;
            border-radius:12px;
            border:1px solid #dbe4ee;
        }

        .badge-row{
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            margin-top:18px;
        }

        .badge{
            background:#eaf4ff;
            color:#0f4c81;
            border:1px solid #cfe4f8;
            padding:8px 12px;
            border-radius:999px;
            font-size:13px;
            font-weight:700;
        }

        .footer-note{
            margin-top:18px;
            color:#64748b;
            font-size:13px;
        }

        @media (max-width: 768px){
            .hero h1{
                font-size:26px;
            }

            .panel{
                padding:18px;
            }

            .upload-box{
                padding:18px;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="hero">
            <h1>Water Segmentation Dashboard</h1>
            <p>
                Upload a TIFF image.
            </p>
        </div>

        <div class="panel">
            <form method="POST" enctype="multipart/form-data">
                <div class="upload-box">
                    <div class="upload-title">Upload satellite image</div>
                    <div class="upload-sub">
                        Supported formats: <code>.tif</code> and <code>.tiff</code>
                    </div>

                    <div class="file-row">
                        <input type="file" name="image_file" accept=".tif,.tiff" required>
                        <button class="btn" type="submit">Run Prediction</button>
                    </div>
                </div>
            </form>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            {% if result %}
                <div class="meta-grid">
                    <div class="metric">
                        <div class="label">Image Name</div>
                        <div class="value">{{ result.file_name }}</div>
                    </div>
                </div>

                <div class="section-title">Prediction Results</div>

                <div class="cards">
                    <div class="card">
                        <h3 class="card-title">RGB Image</h3>
                        <div class="image-wrap">
                            <img src="data:image/png;base64,{{ result.rgb }}" alt="RGB Image">
                        </div>
                    </div>

                    <div class="card">
                        <h3 class="card-title">Predicted Mask</h3>
                        <div class="image-wrap">
                            <img src="data:image/png;base64,{{ result.pred }}" alt="Predicted Mask">
                        </div>
                    </div>

                    <div class="card">
                        <h3 class="card-title">Overlay</h3>
                        <div class="image-wrap">
                            <img src="data:image/png;base64,{{ result.overlay }}" alt="Overlay">
                        </div>
                    </div>
                </div>

                <div class="footer-note">
                    RGB preview is generated from channels [3, 2, 1], and the overlay highlights predicted water regions in red.
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

def normalize_image(image):
    image = image.astype(np.float32).copy()
    for i in range(image.shape[0]):
        band = image[i]
        band = np.clip(band, P_LOW[i], P_HIGH[i])
        band = (band - P_LOW[i]) / (P_HIGH[i] - P_LOW[i] + 1e-6)
        image[i] = band
    return image

def to_base64(img):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_rgb(image_hwc):
    rgb = image_hwc[:, :, [3, 2, 1]].copy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb

def make_overlay(rgb, mask):
    out = rgb.copy()
    red = np.zeros_like(rgb, dtype=np.uint8)
    red[:, :, 0] = mask
    out = cv2.addWeighted(out, 1.0, red, 0.4, 0)
    return out

def read_uploaded_tif(file_storage):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        file_storage.save(tmp.name)
        temp_path = tmp.name

    try:
        with rasterio.open(temp_path) as src:
            img = src.read().astype(np.float32)  # (C,H,W)

        if img.shape[0] != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_CHANNELS} channels, but got {img.shape[0]} channels."
            )

        img = normalize_image(img)
        img = np.transpose(img, (1, 2, 0))  # (H,W,C)
        return img.astype(np.float32)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template_string(HTML, result=None, error=None)

    try:
        if "image_file" not in request.files:
            return render_template_string(HTML, result=None, error="Please upload an image.")

        image_file = request.files["image_file"]

        if image_file.filename == "":
            return render_template_string(HTML, result=None, error="Please choose a TIFF image.")

        img = read_uploaded_tif(image_file)
        rgb = make_rgb(img)

        x = np.expand_dims(img, axis=0)
        pred = model.predict(x, verbose=0)[0, :, :, 0]

        pred_mask = (pred > THRESHOLD).astype(np.uint8) * 255
        ov = make_overlay(rgb, pred_mask)

        result = {
            "file_name": image_file.filename,
            "input_shape": str(x.shape),
            "pred_min": f"{float(pred.min()):.6f}",
            "pred_max": f"{float(pred.max()):.6f}",
            "pred_mean": f"{float(pred.mean()):.6f}",
            "threshold": f"{THRESHOLD:.2f}",
            "rgb": to_base64(rgb),
            "pred": to_base64(pred_mask),
            "overlay": to_base64(ov),
        }

        return render_template_string(HTML, result=result, error=None)

    except Exception as e:
        return render_template_string(HTML, result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)