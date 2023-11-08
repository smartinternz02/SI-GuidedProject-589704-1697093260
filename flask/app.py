from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/Prediction')
def Prediction():
    return render_template('caption.html')

@app.route('/PredictCaption', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(filepath)
        text = modelpredict(filepath)
        return text

def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        # in_text += ' ' + word
        if word == 'endseq':
            break
        in_text += ' ' + word
        # Remove 'startseq' from the beginning and 'endseq' from the end
    in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
    return in_text

def modelpredict(filepath):
    tokenizer = load(open("tokenizer.pkl", 'rb'))
    max_length = 34
    model = load_model("caption.h5")
    photo = extract_features(filepath)
    description = generate_desc(model, tokenizer, photo, max_length)
    return description

if __name__ == "__main__":
    app.run(debug=True)

