# Brain-Tumor-Classification-using-CNN
This project is a web application for classifying brain tumors using a Convolutional Neural Network (CNN). The application is built with Flask for the backend and HTML/CSS for the frontend. Users can upload MRI images of the brain, and the model will predict whether the image indicates the presence of three different types of tumor or not.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [Contact](#contact)

## Introduction
This project aims to provide an easy-to-use interface for medical professionals and researchers to classify brain tumors from MRI images. The underlying model is a CNN, which has been trained on a labeled dataset of brain MRI images.

### User Interface
![Home Page](Flask App/images/homepage.png)
![Login Page](Flask App/images/loginpage.png)
![Classification Page](Flask App/images/classificationpage.png)

## Features
- Upload MRI images for classification.
- Get predictions on the presence of brain tumors.
- Three different types of tumor detection
- Simple and intuitive web interface.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sher-gaha/Brain-Tumor-Classification-using-CNN.git
    cd Brain-Tumor-Classification-using-CNN/Flask App
    ```

2. **Create a virtual environment:**
    ```bash
    conda create -n venv -y 
    conda activate venv
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure you have Flask installed:**
    ```bash
    pip install Flask
    ```

5. **Download the model weights and place them in the appropriate directory (if applicable).**

5. **Download the dataset from kaggle.**
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Usage
To run the application, use the following command in your terminal:
```bash
python -m flask run
```

## Contributors
- Sher Bahadur Gaha 
- Bimal Khamcha (Contributor)

## Contact
For any queries or feedback, please contact: 
- bzugaha55@gmail.com 
- bimalkhamcha2057@gmail.com

