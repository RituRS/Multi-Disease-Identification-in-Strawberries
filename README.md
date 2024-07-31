# Strawberry Disease Classification Web App

## Overview

This project provides a web app for classifying strawberry diseases using deep learning models. The app allows users to upload images of diseased strawberries and get predictions along with confidence scores.

## Requirements

To run the app, you need to have Python installed along with the necessary packages. You can install them using `requirements.txt`.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/yourrepository.git

2. **Navigate to the Project Directory:**
   cd yourrepository

3. **Install Dependencies:**
   pip install -r requirements.txt

4. **Download the Dataset:**
   Dataset Link: [Download Strawberry Disease Detection Dataset from Kaggle](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)
  **Setup Instructions:**
    Go to the provided Kaggle dataset link.
    Download the dataset files.
    Extract the files and place them in the data/ directory within the project folder. Ensure the structure is as follows:
       data/
        train/
        validation/
   The dataset does not use the test severity levels or test subdirectories. Instead, the training set has been split into an 80:20 train-test ratio. Ensure your data is organized accordingly.

5. **Run the app:**
    streamlit run app.py

Ensure that you have downloaded and organized the dataset as specified in the Setup Instructions.

## Model Performance Metrics
Precision, Recall, F1 Score, Coverage, and Misclassification Rate for each model can be viewed within the app.

## Contributing
Feel free to fork the repository and submit pull requests with improvements.
  
