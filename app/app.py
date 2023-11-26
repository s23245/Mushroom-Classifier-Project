import joblib
from PIL import Image
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

model = joblib.load('../updated_model_classifier.pkl')

class_mapping = {
    'Agaricus': 0,
    'Amanita': 1,
    'Boletus': 2,
    'Cortinarius': 3,
    'Entoloma': 4,
    'Hygrocybe': 5,
    'Lactarius': 6,
    'Russula': 7,
    'Suillus': 8,
}


def preprocess_image(file):
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    img_array = preprocess_image(file)

    predictions = model.predict(img_array)

    result = f"Prediction: {predictions}"

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
