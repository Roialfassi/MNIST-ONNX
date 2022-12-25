import onnxruntime
from tensorflow import keras
import onnxmltools
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the input features
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Build and compile the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Convert the model to ONNX
onnx_model = onnxmltools.convert_keras(model)

# Save the ONNX model to a file
onnxmltools.save_model(onnx_model, 'mnist.onnx')

# Load the model
session = onnxruntime.InferenceSession('mnist.onnx')

# Get the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference on some input data
input_data = x_test[0].astype(np.float32).reshape(1, 28, 28)  # use a single test sample as input
output_data = session.run([output_name], {input_name: input_data})[0]

# The output is a probability distribution over the 10 classes
print(output_data)

# Get the predicted class
predicted_class = int(output_data.argmax())
print('Predicted class:', predicted_class)
