# ============================================
# app.py
# Professional Knee Arthritis Detection System
# Flask + VGG19 + ViT UI/UX Template
# ============================================

import os
import sqlite3
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# ============================================
# Flask Config
# ============================================

app = Flask(__name__)
app.secret_key = "arthritis_secret_key"

UPLOAD_FOLDER = "static/uploads"
GRAPH_FOLDER = "static/graphs"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# ============================================
# Database
# ============================================

DB_NAME = "arthritis.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ============================================
# Load Deep Learning Model
# ============================================

MODEL_PATH = "model/arthritis_model.h5"

# Replace with your trained hybrid model
model = load_model(MODEL_PATH)

# ============================================
# Arthritis Classes
# ============================================

classes = [
    "Doubtful",
    "Healthy",
    "Mild",
    "Moderate",
    "Severe"
]

# ============================================
# Home Page
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================
# Upload and Prediction
# ============================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        if 'file' not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files['file']

        if file.filename == "":
            flash("Please select image")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        # ====================================
        # Image Preprocessing
        # ====================================

        img = image.load_img(filepath, target_size=(160, 160))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ====================================
        # Prediction
        # ====================================

        prediction = model.predict(img_array)[0]

        predicted_index = np.argmax(prediction)

        predicted_class = classes[predicted_index]

        confidence = round(float(prediction[predicted_index]) * 100, 2)

        # ====================================
        # Generate Probability Graph
        # ====================================

        plt.figure(figsize=(8, 5))
        plt.bar(classes, prediction)
        plt.xlabel("Classes")
        plt.ylabel("Probability")
        plt.title("Arthritis Prediction Probability")

        graph_path = os.path.join(
            GRAPH_FOLDER,
            "graph.png"
        )

        plt.savefig(graph_path)
        plt.close()

        # ====================================
        # Store in Database
        # ====================================

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions
        (filename, prediction, confidence, date)
        VALUES (?, ?, ?, ?)
        """, (
            filename,
            predicted_class,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            filename=filename,
            prediction=predicted_class,
            confidence=confidence,
            graph="graphs/graph.png"
        )

    return render_template("upload.html")

# ============================================
# History Page
# ============================================

@app.route("/history")
def history():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")

    data = cursor.fetchall()

    conn.close()

    return render_template("history.html", data=data)

# ============================================
# Run Flask App
# ============================================

if __name__ == "__main__":
    app.run(debug=True)