from flask import Flask, jsonify
from tensorflow import keras
import numpy as np

app = Flask(__name__)
model = keras.models.load_model("./assets/inception_200_100epoch.h5")


@app.route('/predict', methods=["GET"])
def predict():
    img = keras.preprocessing.image.load_img("./assets/data/0f6e645466a2.png", target_size=(224, 224))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = model.predict_generator(images)
    if prediction > 0.5:
        predicted = 1
    else:
        predicted = 0
    return jsonify({"prediction": str(predicted)})


if __name__ == '__main__':
    app.run()
