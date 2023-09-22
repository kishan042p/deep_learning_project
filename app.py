from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import werkzeug
from keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps  # Install pillow instead of PIL

app = Flask(__name__)

@app.route('/parkinson',methods=["POST"])
def parkinson():
    if(request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("Deep_learning/uploadedimage/" + filename)

        print(filename)

        np.set_printoptions(suppress=True)
        
        model = load_model("Deep_learning/parkinson.h5", compile=False)


        class_names = open("Deep_learning/label.txt", "r").readlines()


        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


        image = Image.open("Deep_learning/uploadedimage/" + filename).convert("RGB")


        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
        image_array = np.asarray(image)

# Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
        data[0] = normalized_image_array

# Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)


    return jsonify({
            "The Person is" : class_name[2:]
    })
    
@app.route('/malaria',methods=["POST"])
def malaria():
    if(request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("Deep_learning/uploadedimage/" + filename)

        print(filename)

        np.set_printoptions(suppress=True)
        
        model = load_model("Deep_learning/malaria.h5", compile=False)

# Load the labels
        class_names = open("Deep_learning/labels.txt", "r").readlines()


        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
        image = Image.open("Deep_learning/uploadedimage/" + filename).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
        image_array = np.asarray(image)

# Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
        data[0] = normalized_image_array

# Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)


    return jsonify({
            "The Person is" : class_name[2:]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)