# Imports
import gradio as gr
import tensorflow as tf
import numpy as np
import logging

from lime import lime_image
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from keras.models import load_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load the saved model
model = load_model("melanoma_files/model_final.keras")


# Define the labels
labels = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

################################################
#Image classifier
################################################

def classify_image(img_path):
    try:
        # Resize and preprocess the image
        img_resized = tf.image.resize(img_path, (180, 180))
        img = np.expand_dims(img_resized, axis=0)

        # Make a prediction
        prediction = model.predict(img).flatten()
        logging.info(f"Prediction made: {prediction}")

        # Return the top labels with their probabilities
        return {labels[i]: float(prediction[i]) for i in range(len(labels))}
    except Exception as e:
        logging.error(f"Error during image classification: {e}")
        return {label: 0.0 for label in labels}  # Return zero probabilities on error


#Conformal prediction

x= np.load('melanoma_files/tensor_x.npy')
y = np.load('melanoma_files/tensor_y.npy')

# Convert to a TensorFlow tensor
x_calib = tf.convert_to_tensor(x)
# Convert to a TensorFlow tensor
y_calib = tf.convert_to_tensor(y)

def compute_nonconformity(model, x_calib, y_calib):
    # Get probabilities from the model
    probs = model.predict(x_calib)
    # Get the true class probabilities
    class_prob = np.array([probs[i, y_calib[i]] for i in range(len(y_calib))])
    
    return class_prob

# Calculate nonconformity scores
class_probs = compute_nonconformity(model, x_calib, y_calib)

def conformal_prediction(model, x_test, calib_scores, conf_level=90):
    quantile = np.quantile(calib_scores, 1-(int(conf_level)/100))
    # Get softmax probabilities for the test image
    probs_test = model.predict(x_test)
    prediction_sets = [np.where(probs_test[i] >= quantile)[0] for i in range(len(x_test))]
    return prediction_sets


################################################
# LIME
################################################

# Create a LIME image explainer
explainer = lime_image.LimeImageExplainer()

def lime(image):
    
    img_resized = tf.image.resize(image, (180, 180))
    img = np.expand_dims(img_resized, axis=0)

    test_image= tf.reshape(img, (180,180,3))
    test_image = test_image.numpy()
    test_image = test_image.astype('float32') / 255.0 
    predicted_class = model.predict(np.expand_dims(test_image, axis=0)).argmax()
    # Explain the prediction
    explanation = explainer.explain_instance(test_image, model.predict, top_labels=5, hide_color=255, num_samples=100)

    # Get the image and mask
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10,  hide_rest=True)
    return mark_boundaries(temp / 2 + 0.5, mask) 

################################################
# Gradio interface setup
################################################

def process_image_and_float(image, value):

    img_resized = tf.image.resize(image, (180, 180))
    img = np.expand_dims(img_resized, axis=0)
    confidence_set = conformal_prediction(model, img, class_probs, alpha=value)
    classes =[]
    for i in confidence_set[0]:
        classes.append(labels[i])
    return classes or "None"


def validate_input(text):
    try:
        # Convert input to an integer
        value = int(text)
        # Check if the value is between 1 and 10
        if 1 <= value <= 100:
            return True, value
        else:
            return False, "Invalid input. Please enter a number between 1 and 10."
    except ValueError:
        return False, "Invalid input. Please enter a number between 1 and 10."


# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Classification with conformal prediction and LIME")
    # Shared image input for all rows
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload or Select Image", type="pil")
            example_images = ["melanoma_files/sample_images/ISIC_0000002.jpg","melanoma_files/sample_images/ISIC_0000004.jpg","melanoma_files/sample_images/ISIC_0000013.jpg"]

        # Add examples to the image input
            gr.Examples(examples=example_images, inputs=image_input)

    # Row 1: Image input to label
        with gr.Column():
            classify_button = gr.Button("Classify")
            label_output = gr.Label(num_top_classes=3, label="Top Classes")
            classify_button.click(classify_image, inputs=image_input, outputs=label_output)
    
    with gr.Row():
        gr.Markdown("## Conformal prediction")
    # Row 2: Image input and text input to text output
    with gr.Row():
        
        text_input = gr.Textbox(label="Enter a confidence number between 1 and 100",scale=1)
        generate_button = gr.Button("Get classes",scale =0.5)
        text_output = gr.Textbox(label="Generated classes are",scale=3)

    def handle_generate(image, text):
        is_valid, message = validate_input(text)
        if not is_valid:
            return message  # Return the error message to the output
        else:
            # Call your processing function if input is valid
            return process_image_and_float(image, text)

    generate_button.click(handle_generate, inputs=[image_input, text_input], outputs=text_output)    
    
    with gr.Row():
        gr.Markdown("## LIME")

    # Row 3: Image input to image output
    with gr.Row():
        transform_button = gr.Button("Get output")

    with gr.Row():
        image_output = gr.Image(label="Influential parts of the image", type="pil")
        transform_button.click(lime, inputs=image_input, outputs=image_output)


# Launch the interface
demo.launch()

