import os
import time
from flask import Flask, request, jsonify, render_template

from flask_cors import CORS, cross_origin
from firebase_admin import credentials, firestore, initialize_app
import torch
import numpy as np
from joblib import load

import json
import yaml

from PIL import Image

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
#CORS(app, resources={r"/*": {"origins": "*"}})
#cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Firestore DB
cred = credentials.Certificate('src/key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('skin-lesions')


def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)


@app.route('/process', methods=['POST'])
@cross_origin()
def process():
    upload_dir = "images"
    file_names = []
    for key in request.files:
        app.logger.info(key)
        file = request.files[key]
        picture_fn = file.filename
        file_names.append(picture_fn)
        picture_path = os.path.join(upload_dir, picture_fn)
        try:
            file.save(picture_path)
        except:
            print("save fail: " + picture_path)
    app.logger.info(key)
    return json.dumps({"filename": [f for f in file_names]})


@app.route('/saveimage', methods=['POST'])
@cross_origin()
def upload_test():
    upload_dir = "images"
    file_names = []
    for key in request.files:
        app.logger.info(file_names)
        file = request.files[key]
        picture_fn = file.filename
        file_names.append(picture_fn)
        picture_path = os.path.join(upload_dir, picture_fn)
        try:
            file.save(picture_path)
        except:
            print("save fail: " + picture_path)
    app.logger.info(file_names)
    return json.dumps({"filename": [f for f in file_names]})


@app.route('/')
@cross_origin()
def index():
    context = {'server_time': format_server_time()}
    return render_template('index.html', context=context)


@app.route('/add', methods=['POST'])
@cross_origin()
def create():
    """
        create() : Add document to Firestore collection with request body.
        Ensure you pass a custom ID as part of json body in post request,
        e.g. json={'id': '1', 'title': 'Write a blog post'}
    """
    try:
        id = request.json['id']
        todo_ref.document(id).set(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occurred: {e}"


@app.route("/predict_test", methods=['POST'])
@cross_origin()
def predict_test():

    try:
        # Load torch model
        model = torch.jit.load('models/model.zip', map_location='cpu')

        # load image
        img = Image.open(request.files['file'].stream).convert(
            'RGB').resize((456, 456))
        img = np.array(img)
        img = torch.FloatTensor(img.transpose((2, 0, 1)) / 255)

        # get predictions
        preds = model(img.unsqueeze(0)).squeeze()
        probas = torch.softmax(preds, axis=0)

        # apply rescaling
        weights = torch.Tensor([2.1848928544192785, 3.1910118616695438, 11.348979996038821, 17.39556769884639,
                                20.406339031339034, 67.6517119244392, 91.24363057324841, 226.48616600790515, 239.7531380753138])
        new_probas = torch.mul(probas, weights)
        sum_prob = torch.sum(new_probas, dim=0)
        thresholding = torch.div(new_probas, sum_prob)

        ix = torch.argmax(thresholding, axis=0)

        labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']

        return {
            'label': labels[ix],
            'score': thresholding[ix].item(),
        }

    except Exception as e:
        return f"An Error Occurred: {e}"


@app.route("/test", methods=['POST'])
@cross_origin()
def test():
    try:
        age = request.json['age']
        sex = request.json['sex']
        location = request.json['location']
        print("age", age)
        print("sex", sex)
        print("location", location)
        return {
            'age': age,
            'sex': sex,
            'location': location,
        }
    except Exception as e:
        return f"An Error Occurred: {e}"


@app.route("/testmeta", methods=['POST'])
@cross_origin()
def testmeta():
    try:
        # file = request.files['file'].stream
        # file = request.json['file']
        file = request.files['file'].stream

        age = request.form.get('age')
        sex = request.form.get('sex')
        location = request.form.get('location')
        # age = request.json['age']
        # sex = request.json['sex']
        # location = request.json['location']

        # load metadata encoder
        # hot_enconder = load('hot_enconder.joblib')
        # meta = [age, sex, location]
        # onehot = hot_enconder.transform(
        #    [meta]).toarray()
        return {
            'age': age,
            'sex': sex,
            'location': location,

        }
    except Exception as e:
        return f"An Error Occurred: {e}"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():

    try:
        # load image
        img = Image.open(request.files['files'].stream).convert(
            'RGB').resize((380, 380))

        img = np.array(img)
        img = torch.FloatTensor(img.transpose((2, 0, 1)) / 255)

        # handle if server recieves metadata
        try:
            # load meta
            data = request.form.get('files')
            data = yaml.load(data, yaml.SafeLoader)
            app.logger.info(data)
            app.logger.info(data['age'])
            age = data['age']
            sex = data['sex']
            location = data['location']
            meta = [age, sex, location]
            # Load torch model
            model = torch.jit.load('models/model_meta.zip', map_location='cpu')
            model.eval()
            # load metadata encoder
            hot_enconder = load('hot_enconder.joblib')
            meta = hot_enconder.transform(
                [meta]).toarray()
            # sample
            # meta = hot_enconder.transform(
            #    [['30', 'male', 'torso']]).toarray()

            meta = torch.tensor(meta).float()
            app.logger.info("try")

            app.logger.info(meta)

            # get predictions
            preds = model(img.unsqueeze(0), meta.float()).squeeze()
            app.logger.info("meta")
        except:
            # Load torch model
            model = torch.jit.load('models/model.zip', map_location='cpu')
            model.eval()
            # get predictions
            preds = model(img.unsqueeze(0)).squeeze()
            app.logger.info("no meta")
            pass
        app.logger.info("continue")

        probas = torch.softmax(preds, axis=0)

        # apply rescaling
        weights = torch.Tensor([2.1848928544192785, 3.1910118616695438, 11.348979996038821, 17.39556769884639,
                                20.406339031339034, 67.6517119244392, 91.24363057324841, 226.48616600790515, 239.7531380753138])
        new_probas = torch.mul(probas, weights)
        sum_prob = torch.sum(new_probas, dim=0)
        thresholding = torch.div(new_probas, sum_prob)

        ix = torch.argmax(thresholding, axis=0)

        labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']
        classes = [
            "Actinickeratosis",
            "Basal cell carcinoma",
            "Benign keratosis",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic nevus",
            "Squamous cell carcinoma",
            "Unknown",
            "Vascular lesion",
        ]

        app.logger.info(labels[ix])
        return {
            'label': labels[ix],
            'diagnosis': classes[ix],
            'pred': thresholding.tolist(),
            'score': thresholding[ix].item(),
        }

    except Exception as e:
        return f"An Error Occurred: {e}"


port = int(os.environ.get('PORT', 8080))

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=port)
