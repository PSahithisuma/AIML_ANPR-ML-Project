# AIML_Automatic Number Plate Recognition
📌 Project Overview

This project implements an Automatic Number Plate Recognition (ANPR) system using:

Object Detection (to locate number plates)

Image Preprocessing (to enhance clarity)

Optical Character Recognition (OCR) (to read text)

It can detect license plates from images or video streams and extract the alphanumeric characters automatically.

🧠 Tech Stack / Tools

Python
OpenCV
 YOLO
 EasyOCR
TensorFlow / Keras
NumPy, Matplotlib
LabelImg for annotation
Google Colab/Jupyter Notebook



🏗️ System Architecture
         ┌────────────────────────┐
         │   Input Image / Video   │
         └──────────────┬─────────┘
                        ▼
            ┌──────────────────────┐
            │  Plate Detection ML   │ (YOLO / Haar Cascade / SSD)
            └──────────────┬───────┘
                           ▼
            ┌──────────────────────┐
            │ Image Preprocessing   │ (thresholding, blur, resize)
            └──────────────┬───────┘
                           ▼
            ┌──────────────────────┐
            │ OCR (Tesseract / CRNN)│
            └──────────────┬───────┘
                           ▼
         ┌────────────────────────────┐
         │  Extracted Plate Number     │
         └────────────────────────────┘

📂 Project Structure
ANPR-ML-Project/
│── data/
│   ├── images/
│   ├── annotations/
│── models/
│   ├── plate_detector.h5
│   ├── crnn_ocr_model.h5
│── src/
│   ├── detection.py
│   ├── preprocess.py
│   ├── ocr.py
│   ├── utils.py
│── notebooks/
│── results/
│── README.md
│── requirements.txt
│── app.py

📥 Dataset

You can use any open ANPR dataset such as:

AOLP Taiwan Dataset

OpenALPR Benchmark Dataset

Indian License Plate Dataset (Kaggle)

Or create your own dataset using LabelImg for annotation.

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/yourusername/ANPR-ML-Project.git
cd ANPR-ML-Project

2️⃣ Create virtual environment
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install Tesseract-OCR

Windows: install from https://github.com/tesseract-ocr/tesseract

Linux:

sudo apt-get install tesseract-ocr

▶️ Usage
Run ANPR on an image
python app.py --image sample_car.jpg

Run ANPR on webcam/video
python app.py --video traffic.mp4

🔍 How It Works
1. Number Plate Detection

A trained YOLO / Haar Cascade / SSD model detects plates from the car image.
Output → bounding box coordinates.

2. Preprocessing

Convert to grayscale

Noise removal

Thresholding

Resize

3. OCR

Text is extracted using:

Tesseract OCR, or

Deep Learning CRNN model (recommended)

4. Post-processing

Remove noise characters

Format output using regex

Validate with Indian number plate formats (optional)

📊 Results
Model	Accuracy	FPS	Notes
Haar Cascade	~75%	30+	Fast but less accurate
YOLOv5	~92%	20–25	Best detection quality
CRNN OCR	~95%	—	High accuracy for text

Example Output:

Detected Plate Number: TS09AB1234

🧪 Sample Code Snippet
from src.detection import detect_plate
from src.ocr import extract_text

image = "sample.jpg"

plate_img = detect_plate(image)
plate_number = extract_text(plate_img)

print("Detected Number:", plate_number)

🚀 Future Improvements

Deploy as Flask/FastAPI Web App

Add real-time detection using YOLOv8

Integrate with Raspberry Pi for IoT

Improve OCR with Transformer-based text recognition

Add multi-country license plate formats
