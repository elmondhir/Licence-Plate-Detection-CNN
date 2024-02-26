from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.models import load_model  # Assuming you have a Keras model saved
import numpy as np
# Create your views here.

# Load your Keras model
model = load_model(r'C:\Users\Azus\Desktop\CNN-Licence-Plate\Licence-Plate-Detection-CNN\model.h5')

def home(request):
    processed_result = None

    if request.method == 'POST':
        # print("hello world")
        # Assuming you have a form with a file input named 'image-input'
        uploaded_file = request.FILES.get('image-input')
        # uploaded_file = request.FILES['image-input']
        # print(request.FILES)
        # print(request.POST)
        # print(uploaded_file)

        if uploaded_file:
            processed_result = process_image(uploaded_file)

        return render(request, 'home.html', {'processed_result': processed_result})

    return render(request, 'home.html')#

def process_image(uploaded_file):
    # Open the image using Pillow
    image = Image.open(uploaded_file)

    # Convert the original image to a format compatible with your Keras model
    processed_image, scale_factor = preprocess_image(uploaded_file)

    # Use your Keras model for inference and obtain predictions
    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    # Scale the coordinates to the original image size
    scaled_coordinates = scale_coordinates(predictions[0], scale_factor)
    
    return processed_result