from tensorflow.keras.preprocessing import image
import os
from flask import Flask, request, send_from_directory, render_template, make_response, jsonify, Response
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
# import cv2

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'images'
cam = cv2.VideoCapture(0)


# Method to return only allowed file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# Method to run prediction on model then return the result
def predict(file):
    global prediction
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


# def image_predict():
#     success, webcam_img = cam.read()
#     cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'img_test.jpg'), webcam_img)
#
#     response = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'img_test.jpg'))
#     print(response)
#     return response


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Home page
@app.route("/")
def home():
    return render_template('index.html', imagesource='file://null')


# POST request to save image then make prediction on image
@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            response = make_response(jsonify(predict(file_path)), 201)
            return response
    else:
        response = make_response(jsonify({"error": "Method not allowed"}), 405)
        return response


# @app.route('/webcam_feed')
# def video_feed():
#     global cam
#     return Response(image_predict(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)  # debug=False, port=5000, threaded=False
