from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.models import load_model  # Assuming you have a Keras model saved

from PIL import Image, ImageDraw
import numpy as np
# Create your views here.
from django.core.files.uploadedfile import InMemoryUploadedFile


# Load your Keras model
model = load_model(r'C:\Users\Azus\Desktop\CNN-Licence-Plate\Licence-Plate-Detection-CNN\model.h5')
# 
def home(request):
    processed_result = None

    if request.method == 'POST':

        uploaded_file = request.FILES.get('image-input')        

        if uploaded_file:
            processed_result = process_image(uploaded_file)

        return render(request, 'home.html', {'processed_result': processed_result})

    return render(request, 'home.html')#

def process_image(uploaded_file):
    
    # Open the image using Pillow
    image = Image.open(uploaded_file)

    # Convert the original image to a format compatible with your Keras model
    processed_image, scale_factor = pre_process_image(image)

    # Use your Keras model for inference and obtain predictions
    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    # Scale the coordinates to the original image size
    scaled_coordinates = scale_coordinates(predictions[0]*255, scale_factor) ##*255 for normalization
    
    draw = ImageDraw.Draw(image)
    draw.rectangle(scaled_coordinates, outline="red", width=2) # draw scalled coordinates on original image

    # Save the processed image to BytesIO buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Encode the image to base64
    processed_result = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return processed_result

def scale_coordinates(coordinates, scale_factor):
    # Scale the coordinates based on the ratio between processed and original image dimensions
    scaled_coordinates = [
        int(coordinates[0] * scale_factor[0]),
        int(coordinates[1] * scale_factor[1]),
        int(coordinates[2] * scale_factor[0]),
        int(coordinates[3] * scale_factor[1]),
    ]
    return scaled_coordinates


def pre_process_image(original_image):
    # Convert to RGB (if the original image has an alpha channel)
    if original_image.mode == 'RGBA':
        original_image = original_image.convert('RGB')

    # Resize the original image to the required input size
    processed_image_size = (224, 224)
    processed_image = original_image.resize(processed_image_size)

    # Convert the processed image to a NumPy array
    processed_image_array = np.array(processed_image)

    # Normalize the processed image to values between 0 and 1
    processed_image_array = processed_image_array / 255.0

    # Calculate the scale factor for coordinates mapping
    scale_factor = (
        original_image.width / processed_image_size[0],
        original_image.height / processed_image_size[1]
    )

    return processed_image_array, scale_factor
