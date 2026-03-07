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
MASKS_DIR = "data/masks"

EXPECTED_CHANNELS = 12
THRESHOLD = 0.5

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(PERCENTILES_PATH):
    raise FileNotFoundError(f"Percentiles file not found: {PERCENTILES_PATH}")

if not os.path.exists(MASKS_DIR):
    raise FileNotFoundError(f"Masks directory not found: {MASKS_DIR}")

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
        body{
            font-family:Arial,sans-serif;
            background:#f4f7fb;
            padding:30px;
            margin:0;
            color:#222;
        }
        .container{
            max-width:1400px;
            margin:auto;
            background:white;
            padding:25px;
            border-radius:16px;
            box-shadow:0 8px 30px rgba(0,0,0,.08);
        }
        h2{
            margin-top:0;
            color:#0f4c81;
        }
        .note{
            background:#eef6ff;
            border-left:4px solid #0f4c81;
            padding:12px;
            border-radius:10px;
            margin-bottom:20px;
            line-height:1.7;
        }
        .meta{
            margin-top:18px;
            background:#fafafa;
            padding:14px;
            border-radius:12px;
            line-height:1.8;
        }
        .row{
            display:flex;
            gap:14px;
            justify-content:center;
            align-items:flex-start;
            flex-wrap:nowrap;
            overflow-x:auto;
            margin-top:20px;
            padding-bottom:10px;
        }
        .card{
            background:#fafafa;
            padding:12px;
            border-radius:12px;
            box-shadow: inset 0 0 0 1px #eee;
            min-width:240px;
            text-align:center;
            flex:0 0 auto;
        }
        .card h3{
            margin-top:0;
            font-size:18px;
        }
        img{
            width:220px;
            height:auto;
            border-radius:8px;
            border:1px solid #ddd;
        }
        button{
            background:#0f4c81;
            color:white;
            border:none;
            padding:12px 18px;
            border-radius:10px;
            cursor:pointer;
            font-size:15px;
            margin-top:10px;
        }
        button:hover{
            background:#0c3d67;
        }
        .error{
            margin-top:18px;
            background:#ffe9e9;
            color:#9d0000;
            padding:12px;
            border-radius:10px;
        }
        code{
            background:#f1f1f1;
            padding:2px 6px;
            border-radius:6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Water Segmentation</h2>

        <form method="POST" enctype="multipart/form-data">
            <p>Upload Image (.tif)</p>
            <input type="file" name="image_file" accept=".tif,.tiff" required>
            <br><br>
            <button type="submit">Predict</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
            <div class="meta">
                <b>Image name:</b> {{ result.file_name }}<br>
                <b>Ground truth file:</b> {{ result.gt_name }}<br>
            </div>

            <div class="row">
                <div class="card">
                    <h3>Image</h3>
                    <img src="data:image/png;base64,{{ result.rgb }}">
                </div>

                <div class="card">
                    <h3>Ground Truth</h3>
                    <img src="data:image/png;base64,{{ result.gt }}">
                </div>

                <div class="card">
                    <h3>Predicted Mask</h3>
                    <img src="data:image/png;base64,{{ result.pred }}">
                </div>

                <div class="card">
                    <h3>Overlay</h3>
                    <img src="data:image/png;base64,{{ result.overlay }}">
                </div>
            </div>
        {% endif %}
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

def compute_metrics(gt, pred):
    gt_bin = (gt > 0).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)

    tp = np.sum((gt_bin == 1) & (pred_bin == 1))
    fp = np.sum((gt_bin == 0) & (pred_bin == 1))
    fn = np.sum((gt_bin == 1) & (pred_bin == 0))

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return iou, precision, recall, f1

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

def find_ground_truth_mask(image_filename):
    base = os.path.splitext(image_filename)[0]
    candidates = [
        os.path.join(MASKS_DIR, base + ".png"),
        os.path.join(MASKS_DIR, base + ".jpg"),
        os.path.join(MASKS_DIR, base + ".jpeg"),
        os.path.join(MASKS_DIR, base + ".tif"),
        os.path.join(MASKS_DIR, base + ".tiff"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Ground truth mask not found for '{image_filename}'. "
        f"Expected one of: {', '.join(candidates)}"
    )

def read_ground_truth(mask_path, target_hw):
    ext = os.path.splitext(mask_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        with rasterio.open(mask_path) as src:
            gt = src.read(1)
    else:
        gt = Image.open(mask_path)
        gt = np.array(gt)

    if gt.ndim == 3:
        gt = gt[:, :, 0]

    if gt.shape[:2] != target_hw:
        gt = cv2.resize(gt, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

    print("GT unique before binarize:", np.unique(gt)[:20])

    if gt.max() <= 1:
        gt = gt.astype(np.uint8) * 255
    else:
        gt = (gt > 127).astype(np.uint8) * 255

    print("GT unique after binarize:", np.unique(gt))
    return gt

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

        gt_path = find_ground_truth_mask(image_file.filename)
        print("Uploaded image:", image_file.filename)
        print("Matched GT path:", gt_path)

        gt = read_ground_truth(gt_path, img.shape[:2])

        x = np.expand_dims(img, axis=0)
        pred = model.predict(x, verbose=0)[0, :, :, 0]

        pred_mask = (pred > THRESHOLD).astype(np.uint8) * 255
        ov = make_overlay(rgb, pred_mask)

        iou, precision, recall, f1 = compute_metrics(gt, pred_mask)

        result = {
            "file_name": image_file.filename,
            "gt_name": os.path.basename(gt_path),
            "input_shape": str(x.shape),
            "pred_min": f"{float(pred.min()):.6f}",
            "pred_max": f"{float(pred.max()):.6f}",
            "pred_mean": f"{float(pred.mean()):.6f}",
            "threshold": f"{THRESHOLD:.2f}",
            "iou": f"{iou:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1": f"{f1:.4f}",
            "rgb": to_base64(rgb),
            "gt": to_base64(gt),
            "pred": to_base64(pred_mask),
            "overlay": to_base64(ov),
        }

        return render_template_string(HTML, result=result, error=None)

    except Exception as e:
        return render_template_string(HTML, result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)