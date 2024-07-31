import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Caching model loading to improve performance
@st.cache_resource
def load_models():
    model_a = load_model("C:/Users/RITUJA/Strawberry Disease Detection/strawberry_disease_classification_model_a.keras")
    model_b = load_model("C:/Users/RITUJA/Strawberry Disease Detection/strawberry_disease_classification_model_b.keras")
    model_c = load_model("C:/Users/RITUJA/Strawberry Disease Detection/strawberry_disease_classification_model_c.keras")
    model_d = load_model("C:/Users/RITUJA/Strawberry Disease Detection/strawberry_disease_classification_model_d.keras")
    return model_a, model_b, model_c, model_d

model_a, model_b, model_c, model_d = load_models()

# Define class names
class_names = [
    'Angular Leaf Spot',
    'Anthracnose Fruit Rot',
    'Blossom Blight',
    'Gray Mold',
    'Leaf Spot',
    'Powdery Mildew Fruit',
    'Powdery Mildew Leaf'
]

# Define function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define function to classify image
def classify_image(model, image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    return class_names[class_idx], confidence

# Function to show dataset distribution histogram
def show_histogram():
    st.subheader("Dataset Distribution Histogram")

    # Dataset distribution
    original_train_size = 1450
    validation_size = 307
    custom_train_size = 1160
    custom_test_size = 290

    # Data labels and sizes for original and custom splits
    data_labels = ['Original Train Set', 'Validation Set', 'Custom Train Set', 'Custom Test Set']
    data_sizes = [original_train_size, validation_size, custom_train_size, custom_test_size]

    # Plotting the histogram
    plt.figure(figsize=(12, 8))
    bars = plt.bar(data_labels, data_sizes, color=['blue', 'green', 'purple', 'orange'])

    # Setting font properties for labels and title
    plt.xlabel('Dataset Subsets', fontsize=16, fontname='Times New Roman')
    plt.ylabel('Number of Images', fontsize=16, fontname='Times New Roman')
    plt.title('Dataset Distribution', fontsize=16, fontname='Times New Roman')

    # Setting font size for tick labels on x-axis and y-axis
    plt.xticks(fontsize=16, fontname='Times New Roman')
    plt.yticks(fontsize=16, fontname='Times New Roman')

    # Annotating the bars with the respective sizes
    for bar, size in zip(bars, data_sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, 
                 bar.get_height() + 20, 
                 size, 
                 ha='center', 
                 va='bottom', 
                 fontsize=16, 
                 fontname='Times New Roman')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    buf.seek(0)
    st.image(buf, caption='Dataset Distribution Histogram', use_column_width=True)
    plt.close()

# Function to show model performance metrics in a bar chart
def show_metrics(model_name):
    st.subheader(f"{model_name} Performance Metrics")

    # Define metric values based on the selected model
    metrics = {
        "Model A": {"precision": 0.86, "recall": 0.89, "f1_score": 0.87, "coverage": 0.89, "mr": 0.11},
        "Model B": {"precision": 0.83, "recall": 0.87, "f1_score": 0.84, "coverage": 0.87, "mr": 0.13},
        "Model C": {"precision": 0.03, "recall": 0.16, "f1_score": 0.04, "coverage": 0.16, "mr": 0.84},
        "Model D": {"precision": 0.99, "recall": 0.99, "f1_score": 0.99, "coverage": 0.99, "mr": 0.01},
    }

    # Fetch metrics for the selected model
    model_metrics = metrics[model_name]

    # Plot the metrics as a bar chart
    metric_names = list(model_metrics.keys())
    metric_values = list(model_metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color='skyblue')

    # Adding labels and title
    plt.xlabel('Metrics', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Values', fontsize=14, fontname='Times New Roman')
    plt.title(f'{model_name} Performance Metrics', fontsize=16, fontname='Times New Roman')

    # Adding value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontname='Times New Roman')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    st.image(buf, caption=f'{model_name} Performance Metrics', use_column_width=True)
    plt.close()

# Streamlit app
st.title("Strawberry Disease Classification")

# Show histogram button
if st.button('Show Dataset Distribution Histogram'):
    show_histogram()

# Upload image
uploaded_image = st.file_uploader("Choose an image of diseased strawberry", type=["jpg", "png"])

# Select model
model_option = st.selectbox(
    "Select Model",
    ["Model A", "Model B", "Model C", "Model D"]
)

# Classification logic
if uploaded_image is not None:
    # Load image
    image = Image.open(uploaded_image)
    
    # Select the model
    if model_option == "Model A":
        model = model_a
    elif model_option == "Model B":
        model = model_b
    elif model_option == "Model C":
        model = model_c
    elif model_option == "Model D":
        model = model_d

    # Classify the image
    class_name, confidence = classify_image(model, image)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Display classification results
    st.write(f"Prediction: {class_name}")
    st.write(f"Confidence: {confidence:.2f}")

# Show model metrics button
if st.button('Show Model Metrics'):
    show_metrics(model_option)