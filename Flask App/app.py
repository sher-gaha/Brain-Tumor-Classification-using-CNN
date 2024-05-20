from flask import Flask, render_template, request, redirect, url_for, session, get_flashed_messages, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib

from PIL import Image
from keras.utils import img_to_array



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Load the pre-trained TensorFlow model
# model_path = 'brain_tumor_model(inceptionV3)second.h5'
# model = None

#model_filename = 'brain_tumor_model(inceptionV3)second.h5'
#model_path = os.path.join(os.path.dirname(__file__), 'model', model_filename)

model_path = 'C:/Users/Asus/Music/csit 7th sem/Final year project/BrainTumorDetectionFlask-master/model/brain_tumor_model(inceptionV3)second.h5'

# Load the pre-trained TensorFlow model
#model = None

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("Warning: Model not found!")

# Dummy user data for login (replace with actual user authentication logic)
users = [
    {'username': 'sher', 'password': 'sher123'},
    {'username': 'bimal', 'password': 'bimal123'},
]

# Dummy classification results (replace with actual classification logic)
classification_results = {
    'glioma': 'Glioma tumor',
    'meningioma': 'Meningioma tumor',
    'pituitary': 'Pituitary tumor',
    'no_tumor': 'No tumor',
}

# Variable to store the current user session
current_user = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html', current_user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    global current_user

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = next((u for u in users if u['username'] == username and u['password'] == password), None)

        if user:
            current_user = user
            return redirect(url_for('classification'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html', error=None)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    global current_user

    if current_user is None:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'selected_image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        selected_image = request.files.get('selected_image')
        #selected_class = request.form.get('selected_class')
        #print("Image Array Values:", selected_image)


        if selected_image.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        # Create the 'uploads' directory if it doesn't exist
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        if selected_image and allowed_file(selected_image.filename):
            filename = secure_filename(selected_image.filename)
            filepath = os.path.join(upload_folder, filename)
            selected_image.save(filepath)

            # Use the loaded TensorFlow model to classify the image
            result = classify_image(filepath)

            os.remove(filepath)  # Remove the uploaded file after classification
            flash(result, 'success')
            return redirect(url_for('classification'))

        flash('Invalid file type', 'error')
        return redirect(url_for('classification'))

    return render_template('classification.html', current_user=current_user)

def classify_image(img_path):

    if model is not None:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label
    else:
        return "model not found"

        

@app.route('/about')
def about():
    return render_template('about.html')

# Route to render the Contact page
@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


@app.route('/abstract')
def abstract():
    # Your logic for the abstract page
    return render_template('abstract.html')

if __name__ == '__main__':
    app.run(debug=True)
