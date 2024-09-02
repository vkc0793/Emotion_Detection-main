from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import cv2
import os
import sqlite3

app = Flask(__name__)

# Load the model
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model("model_a.json", "model_weights.weights.h5")
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Ensure the uploads directory exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            emotion = detect_emotion(file_path)
            return render_template('index.html', emotion=emotion, image=file_path)
    return render_template('index.html')

def detect_emotion(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        return pred
    except:
        return "Unable to detect"


@app.route("/contact", methods=["GET", "POST"])
def contactus():
    if request.method == "POST":
        fname = request.form.get("fullname")
        pno = request.form.get("phone")
        email = request.form.get("email")
        addr = request.form.get("address")
        msg = request.form.get("message")
        conn = sqlite3.connect("emotion.db")
        curr = conn.cursor()
        curr.execute(f'''INSERT INTO CONTACT VALUES("{fname}", "{pno}", "{email}", "{addr}", "{msg}")''')
        conn.commit()
        conn.close()
        return render_template("message.html")
    else:
        return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
