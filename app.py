from flask import Flask, request
import os
import tempfile
import model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:

        return 'No image'

    image = request.files['image']

    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, 'temp_image.jpg')
    image.save(temp_file_path)

    # Perform predictions
    predictions = model.predict(temp_file_path)
    # Remove the temporary file
    os.remove(temp_file_path)
    os.rmdir(temp_dir)

    return {

        "result": predictions

    }
    ###


@app.route('/index')
def index():

    return 'Hello, World!'


if __name__ == '__main__':

    app.run(host='localhost', port=8000)
