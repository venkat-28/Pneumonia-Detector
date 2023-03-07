
from flask import Flask ,render_template,request,send_file
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import cv2
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import os
from werkzeug.utils import secure_filename



model = load_model("trained.h5")




app = Flask(__name__)
UPLOAD_FOLDER = 'static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/result',methods=['GET','POST'])

def submit():
    global prediction,a
    prediction=""
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            print(image.filename)
            img= cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            tempimg = img
            img = cv2.resize(img,(300,300))
            img = img/255.0
            img = img.reshape(1,300,300,3)
            a=model.predict(img)
            prediction = model.predict(img) >= 0.5
            if prediction>=0.5:
                prediction = "Pneumonia"
            else:
                prediction = "Normal"
        return render_template('result.html',accuracy_pneumonia=a[0],result_pneumonia=prediction)
    return render_template('result.html')

if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run( debug =True)
