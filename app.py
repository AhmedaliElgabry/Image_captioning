from flask import Flask,flash, request, jsonify, render_template,redirect,send_from_directory,session, url_for
import flask
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
from io import BytesIO
import numpy as np
import json

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#lood deep learning model
caption_model = tf.keras.models.load_model('caption_model.model')

encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

def encodeImage(img):
  # Resize all images to a standard size (specified bythe image encoding network)
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
  # Convert a PIL image to a numpy array
  x = tensorflow.keras.preprocessing.image.img_to_array(img)
  # Expand to 2D array
  x = np.expand_dims(x, axis=0)
  # Perform any preprocessing needed by InceptionV3 or others
  x = preprocess_input(x)
  # Call InceptionV3 (or other) to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image
  # Shape to correct form to be accepted by LSTM captioning network.
  x = np.reshape(x, OUTPUT_DIM )
  return x

wordtoidx = open('wordtoidx.pickle','rb')
wordtoidx = pickle.load(wordtoidx)

idxtoword = open('idxtoword.pickle','rb')
idxtoword = pickle.load(idxtoword)

max_length = 34

def generateCaption(photo):
    in_text = 'START'
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        #print('sequence 1 : ',sequence)
        sequence = pad_sequences([sequence], maxlen=max_length)
        #print('sequence2: ',sequence)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        #print('yhat 1 : ',yhat)
        yhat = np.argmax(yhat)
        #print('yhat 2 : ',yhat)
        word = idxtoword[yhat]
        #print('word : ',word)
        in_text += ' ' + word
        #print('in_text: ',in_text)
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final







UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/posts',methods = ['GET','POST'])
def home():
    post2 = 'ahmed'
    if request.method == 'POST':
      file = request.files['mm']
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      #response = requests.get(url)
      #img = Image.open(BytesIO(file))
      #img.load()
      print('filename : ',filename)
      path = r'./static/{}'.format(filename)

      img = Image.open(path)
      #plt.imshow(img)
      #plt.show()
      
      #response = requests.get(url)

      img = encodeImage(img).reshape((1,OUTPUT_DIM))

      caption = generateCaption(img)

      print("Caption:",caption)
      print("_____________________________________")
      data = {"image_name":filename,"caption":caption}
      return redirect(url_for('success', name=filename,caption=caption))

    else:
      return render_template('index.html')

@app.route('/success/<name>/<caption>')
def success(name,caption):
    print('name: ',name)
    print('caption: ',caption)
    data = {'image_name': name,'caption':caption}
    return render_template('index.html',data = data)

if __name__ == "__main__":
    app.run(debug=True)