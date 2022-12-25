import onnxruntime
import numpy as np
from PIL import Image
import tkinter as tk
import subprocess
from tkinter import filedialog
import tkinter.ttk as ttk


# Define the callback function for the "Open Paint" button
def open_paint():
    subprocess.run(['mspaint'])  # run the Paint app


# Find file of PNG then tests it
def get_file():
    photo_chosen = filedialog.askopenfilename(filetypes=[('PNG files', '*.png')])
    global image_file
    global image_label
    image_file= tk.PhotoImage(file=photo_chosen)
    image_label = tk.Label(image=image_file)
    image_label.pack()
    test_model(photo_chosen)



# Define the callback function for the "Exit" button
def exit_program():
    root.destroy()  # destroy the tkinter window


def test_model(photo):
    # Load the ONNX model
    session = onnxruntime.InferenceSession('mnist.onnx')

    # Get the input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load the input image and preprocess it
    image = Image.open(photo)
    image = image.convert('L')  # convert to grayscale
    image = image.resize((28, 28))  # resize to 28x28 pixels
    image = np.array(image).astype(np.float32) / 255.0  # normalize and convert to float
    image = image.reshape(1, 28, 28)  # add batch dimension


    # Run inference
    output_data = session.run([output_name], {input_name: image})[0]

    # The output is a probability distribution over the 10 classes
    # print(output_data)

    # Get the predicted class
    predicted_class = int(output_data.argmax())
    label.config(text=str('Predicted class:') + str(predicted_class))


if __name__ == "__main__":
    root = tk.Tk()
    # ttk.Style().theme_use("aqua")
    root.title('Test ONNX MNIST')  # set the title of the main window
    root.geometry("640x480")
    # Create the "Open Paint" button
    open_paint_button = tk.Button(root, text='Open Paint', command=open_paint)
    open_paint_button.pack()

    # Create the "Exit" button
    exit_button = tk.Button(root, text='Test Model On your file', command=get_file)
    exit_button.pack()

    # Create the "Exit" button
    exit_button = tk.Button(root, text='Exit', command=exit_program)
    exit_button.pack()

    image_file = tk.PhotoImage()
    image_label = tk.Label(image=image_file)
    image_label.pack()

    # Pridiction Label
    label = tk.Label(text="Predicted class")
    label.pack()




    # Run the tkinter loop
    root.mainloop()
