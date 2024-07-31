import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd

# Cache model loading with st.cache_resource
@st.cache_resource
def load_models():
    model_a = load_model("strawberry_disease_classification_model_a.keras")
    model_b = load_model("strawberry_disease_classification_model_b.keras")
    model_c = load_model("strawberry_disease_classification_model_c.keras")
    model_d = load_model("strawberry_disease_classification_model_d.keras")
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

# Streamlit app
st.title("Strawberry Disease Classification")

# Display dataset distribution histogram
@st.cache_resource
def get_histogram_image():
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

    # Save the plot to a BytesIO object and return
    image = io.BytesIO()
    plt.savefig(image, format='jpg', dpi=600)
    image.seek(0)
    return image

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

# Define model metrics
model_metrics = {
    "Model A": {
        "Precision": 0.86,
        "Recall": 0.89,
        "F1 Score": 0.87,
        "Coverage": 0.89,
        "Misclassification Rate": 0.11
    },
    "Model B": {
        "Precision": 0.83,
        "Recall": 0.87,
        "F1 Score": 0.84,
        "Coverage": 0.87,
        "Misclassification Rate": 0.13
    },
    "Model C": {
        "Precision": 0.03,
        "Recall": 0.16,
        "F1 Score": 0.04,
        "Coverage": 0.16,
        "Misclassification Rate": 0.84
    },
    "Model D": {
        "Precision": 0.99,
        "Recall": 0.99,
        "F1 Score": 0.99,
        "Coverage": 0.99,
        "Misclassification Rate": 0.01
    }
}

# Show metrics based on selected model
def show_metrics(model_name):
    metrics = model_metrics.get(model_name, {})
    if metrics:
        st.subheader(f"{model_name} Performance Metrics")
        
        # Convert metrics to a DataFrame for plotting
        df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        
        # Plot metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df['Metric'], df['Value'], color='skyblue')
        ax.set_xlabel('Metric', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(f"{model_name} Metrics", fontsize=16)
        
        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom', fontsize=12)
        
        st.pyplot(fig)

# Show histogram and metrics based on user interaction
if st.button('Show Dataset Distribution Histogram'):
    histogram_image = get_histogram_image()
    st.image(histogram_image, caption='Dataset Distribution Histogram')

if st.button('Show Model Metrics'):
    if model_option:
        show_metrics(model_option)
    else:
        st.write("Please select a model to see the metrics.")
