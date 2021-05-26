from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model
from keras.preprocessing import image
import numpy as np
import os
from keras import backend as K
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'images'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    best_model = load_model('best_model_during_training_v1.h5')

    # load an image to be tested (random image from Google)
    img_pred = image.load_img(file, target_size=(200, 200))

    # convert image to numpy array
    img_pred = image.img_to_array(img_pred)
    # expand the array
    img_pred = np.expand_dims(img_pred, axis=0)
    # predict the image from the img_pred
    result = best_model.predict(img_pred)

    if result[0][0] == 1:
        prediction = 'Recyclable'

    elif result[0][0] == 0:
        prediction = 'Organic'

    return prediction


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('index.html', label='', imagesource='file://null')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return output


@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
