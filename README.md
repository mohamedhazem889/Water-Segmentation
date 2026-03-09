🌊 Water Segmentation from Satellite Images

A deep learning project for water body segmentation from multi-spectral satellite imagery using a trained TensorFlow model and a Flask-based web interface.

The system allows users to upload multi-band TIFF satellite images, performs preprocessing and normalization, and predicts water segmentation masks using a trained model.

🚀 Demo

The web interface allows you to:

Upload a .tif / .tiff satellite image

Run the trained segmentation model

Visualize:

• RGB preview
• Predicted water mask
• Overlay visualization

🧠 Model Overview

The model is trained to segment water bodies from 12-channel satellite imagery.

Input

Image size: dynamic

Channels: 12 bands

Processing Pipeline

Read TIFF image using rasterio

Normalize each band using percentile scaling

Convert to (H, W, C) format

Run inference with TensorFlow model

Apply threshold to generate binary mask

Create overlay visualization

📂 Project Structure
project/
│
├── app.py
├── best_model.keras
├── band_percentiles.npz
│
├── Dockerfile
├── requirements.txt
│
├── notebooks/
│   ├── U-net.ipynb
│   └── Pretrained_unet.ipynb
│
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/water-segmentation.git
cd water-segmentation

Install dependencies:

pip install -r requirements.txt

Dependencies used in the project include Flask, TensorFlow, Rasterio, OpenCV, and Pillow. 

requirements

▶️ Run the Application

Start the Flask server:

python app.py

Then open:

http://localhost:5000

The app will load the trained model and allow image upload and prediction. 

app

🖼 Example Output

The system generates three visual outputs:

RGB Preview

Generated from bands [3,2,1]

Predicted Mask

Binary segmentation mask of water areas.

Overlay

Predicted water regions highlighted on the RGB image.

🧪 Prediction Pipeline
pred = model.predict(x)[0,:,:,0]
pred_mask = (pred > THRESHOLD)

Threshold used for segmentation:

THRESHOLD = 0.48
🐳 Docker Deployment

Build the Docker image:

docker build -t water-segmentation .

Run the container:

docker run -p 5000:5000 water-segmentation

Then open:

http://localhost:5000
📊 Input Image Requirements

The uploaded image must:

Be TIFF format

Contain 12 spectral bands

Match the preprocessing used during training

If the image contains a different number of channels, the app will raise an error.

🧩 Key Features

✔ Multi-band satellite image support
✔ Deep learning segmentation model
✔ Automatic normalization
✔ Web-based visualization
✔ Docker deployment ready

🛠 Technologies Used

Python

TensorFlow / Keras

Flask

Rasterio

OpenCV

Docker

📌 Future Improvements

Add support for GeoTIFF metadata

Export segmentation masks as GeoTIFF

Improve visualization and UI

Add model performance metrics
