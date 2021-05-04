from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './assets/data'
app = Flask(__name__)
model = keras.models.load_model("./assets/inception_200_100epoch.h5")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=["POST"])
def upload():
    file = request.files['']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({"Success": "True",
                    "message": "success upload file",
                    "data": {
                        "img_path": str(filename)
                    }}, 201)


@app.route('/predict/<file_name>', methods=["GET"])
def predict(file_name):
    img_path = UPLOAD_FOLDER + '/' + file_name
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = model.predict_generator(images)
    if prediction > 0.5:
        predicted = 1
    else:
        predicted = 0
    result = predicted
    return jsonify({"success": "True",
                    "message": "Success getting prediction",
                    "data": {
                        "prediction": str(result)
                    }}, 200)


if __name__ == '__main__':
    app.run()
