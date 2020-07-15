from flask import Flask, render_template, request ,url_for

from skimage.transform import resize
from imageio import imread
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


#global vars for easy reusability
global model, graph
#initialize these variables
#global model
json_file = open('model_json.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("weights.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#global graph
graph = tf.compat.v1.get_default_graph()




app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

import re
import base64

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    try:
        convertImage(imgData)
    except:
        f = request.files['img']
        f.save('output.png')
   
    x = imread('output.png', pilmode='L')
   
    x = resize(x, (28, 28))
   
    x = x.reshape(1, 28, 28, 1)
    
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        # convert the response to a string
        response = np.argmax(out, axis=1)
        return str(response[0])


if __name__ == "__main__":

    # run the app locally on the given port
    #app.run(host='0.0.0.0', port=80)
	# optional if we want to run in debugging mode
    app.run()
