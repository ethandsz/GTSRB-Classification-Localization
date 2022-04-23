import base64
import numpy as np
import io
from PIL import Image
import random
from matplotlib.image import imread
import matplotlib.patches as patches
import matplotlib.image as mpimg
import cv2
from keras import backend as K
from keras.models import load_model
from flask import request
from flask import jsonify
from flask import Flask, redirect, url_for, render_template
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import img_to_array
import os
from io import BytesIO
from io import StringIO


import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)



def get_model():
    global model
    model = load_model("BBox600EpochModel.h5")
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    #print("* READING")
    #image = cv2.imread(image)
    #print("* FromARray")
    #image_fromarray = Image.fromarray(image, 'RGB')
    #resize_image = image_fromarray.resize((32, 32))
    #print("* RESIZED")
    #return resize_image
    print(" * CONVERTING")
    image = image.convert("RGB")
    global width
    global height
    width, height = image.size
    print(" * CONVERTED")
    global im
    im = (image)
    image = image.resize(size=(32, 32))
    print(" *RESIZED")
    image = img_to_array(image)
    image = np.array(image)
    print("* RESHAPING")
    image = image / 255
    image = image.reshape(-1, 32, 32, 3)
    print(" *FINISHED")
    #print("*" + im)
    return image

print(" * Loading Keras Model...")
get_model()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

@app.route("/predict", methods=["GET", "POST"])
def predict():
    global labels
    labels = {0: 'Speed limit (20km/h)',
              1: 'Speed limit (30km/h)',
              2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)',
              4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)',
              6: 'End of speed limit (80km/h)',
              7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)',
              9: 'No passing',
              10: 'No passing veh over 3.5 tons',
              11: 'Right-of-way at intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Veh > 3.5 tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve left',
              20: 'Dangerous curve right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End speed + passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End no passing veh > 3.5 tons'}
    print(" * PREDICT")
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(32,32))
    print("* PROCESSED")
    prediction = model.predict(processed_image)
    classes_x = np.argmax(prediction[1], axis=1)
    global sign
    sign = (labels[int(classes_x)])
    print("* PREDICTED")
    print("* ", prediction[0] * 32)
    #response = {
    #    'prediction': {
    #        'sign': sign,
    #        'box': "prediction[0]"
    #    }
    #}
    global pred_x1
    global pred_x2
    global pred_y1
    global pred_y2
    pred_x1 = (prediction[0][0][0]) * width
    pred_x2 = (prediction[0][0][1]) * width
    pred_y1 = (prediction[0][0][2]) * height
    pred_y2 = (prediction[0][0][3]) * height
    print(" * ", pred_x1)
    return redirect("/static/plot.html")

    #return jsonify(response)

@app.route('/plot.png')
def plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    #xs = range(100)
    #ys = [random.randint(1, 50) for x in xs]
    axis.imshow(im)
    rect = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1, linewidth=2, edgecolor='b',
                             facecolor='none')
    axis.add_patch(rect)
    text = axis.text(pred_x1, pred_y1 - 1, sign, fontsize=12, color='lime')
    return fig


if __name__ == '__main__':
    app.run(debug=True)





