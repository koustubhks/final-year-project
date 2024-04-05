from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import backend as K
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = 'models/Brain.h5'
model = load_model(model_path)

# Grad-CAM function
def grad_cam(model, img_array, layer_name):
    cls = np.argmax(model.predict(img_array))
    class_output = model.output[:, cls]
    last_conv_layer = model.get_layer(layer_name)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_array])
    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(240, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = imagenet_utils.preprocess_input(img_array)
    return img_array

# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No file included'})

        # Get the uploaded file
        file = request.files['image']

        # Save the uploaded file to a temporary location
        file_path = 'temp.jpg'
        file.save(file_path)

        # Preprocess the uploaded image
        processed_img = preprocess_image(file_path)

        # Make predictions
        prediction = model.predict(processed_img)

        # Convert prediction to human-readable format
        class_dict = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        tumor_type = class_dict[np.argmax(prediction)]

        # Generate heatmap using Grad-CAM
        heatmap = grad_cam(model, processed_img, 'conv2d_1')  # Change 'conv2d_1' to your desired convolutional layer name

        # Return the prediction and heatmap
        return jsonify({'prediction': tumor_type, 'heatmap': heatmap.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
