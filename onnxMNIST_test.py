import onnxruntime
import numpy as np
from PIL import Image

# Load the ONNX model
session = onnxruntime.InferenceSession('mnist.onnx')

# Get the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load the input image and preprocess it
image = Image.open('image.png')
image = image.convert('L')  # convert to grayscale
image = image.resize((28, 28))  # resize to 28x28 pixels
image = np.array(image).astype(np.float32) / 255.0  # normalize and convert to float
image = image.reshape(1, 28, 28)  # add batch dimension

# Run inference
output_data = session.run([output_name], {input_name: image})[0]

# The output is a probability distribution over the 10 classes
print(output_data)

# Get the predicted class
predicted_class = int(output_data.argmax())
print('Predicted class:', predicted_class)
