import os
import time
from flask import Flask, request, jsonify, render_template
from firebase_admin import credentials, firestore, initialize_app
import torch
import numpy as np

import joblib
import pandas as pd

from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load torch model
model = torch.jit.load('models/model.zip', map_location='cpu')

# Initialize Firestore DB
cred = credentials.Certificate('src/key.json')
default_app = initialize_app(cred)
db = firestore.client()
todo_ref = db.collection('skin-lesions')


def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)


@app.route('/')
def index():
    context = {'server_time': format_server_time()}
    return render_template('index.html', context=context)


@app.route('/add', methods=['POST'])
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


@app.route("/predict", methods=['POST'])
def predict():

    try:
        # load image
        img = Image.open(request.files['file'].stream).convert(
            'RGB').resize((456, 456))
        img = np.array(img)
        img = torch.FloatTensor(img.transpose((2, 0, 1)) / 255)

        # load encoded metadata
        # https://stackoverflow.com/questions/59511198/save-and-load-one-hot-encoding-for-ml
        #enc = joblib.load('encoder.joblib')
        #data_df = pd.DataFrame(data=data, columns=col_names)
        #enc_df = pd.DataFrame(data=enc.transform(data).toarray(), columns=enc.get_feature_names(col_names), dtype=bool)
        #df = pd.concat([data_df, enc_df], axis=1)

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


port = int(os.environ.get('PORT', 8080))

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=port)
