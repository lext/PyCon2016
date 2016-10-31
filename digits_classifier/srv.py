from flask import Flask, request, session, g, redirect, url_for, abort, render_template, make_response
from sklearn.externals import joblib
from io import BytesIO
from skimage import io as skio
from skimage.transform import resize

import base64
import numpy as np


app = Flask(__name__, static_url_path='/static')
clf = joblib.load('clf.pkl')


@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/recognizer', methods=['POST'])
def recognize():
    data = request.get_json(silent=True)['image']
    data = data[22:]

    img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]

    img = resize(img, (28, 28))

    number = clf.predict(img.reshape(1, -1))[0]

    return make_response(str(number),200)
