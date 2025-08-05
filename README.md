Breast Cancer Subtype Classification using CNN

This repository contains a deep learning project that classifies breast cancer subtypes from histopathological images using a Convolutional Neural Network (CNN). This project was completed as part of a training program in Bioinformatics and AI/ML.

🔬 Project Overview

Title: Cancer Subtype Identification from Histopathological Images using CNN

Objective: Detect and classify subtypes of breast cancer (benign or malignant) using deep learning techniques.

Dataset: BreakHis dataset (sourced via Kaggle)

Platform: Google Colab (cloud-based environment)

🛠️ Tools & Libraries Used

Python 3.x

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

📊 Methodology

The dataset was split into training and testing sets.

CNN architecture was built using the Keras Sequential API.

The model was trained and evaluated using performance metrics, including accuracy, loss, and a confusion matrix.

Data augmentation and normalization techniques were applied to enhance model generalization.

🔄 Workflow

Data Loading: Uploaded data from Kaggle using kaggle.json

Preprocessing: Resized, normalized, and split images

Modeling: CNN layers with Conv2D, MaxPooling, Dropout, Flatten, Dense

Training: Used categorical crossentropy and Adam optimizer

Evaluation: Accuracy, confusion matrix, and classification report

📁 Repository Contents

breast_cancer_histopathology.ipynb: Main notebook

requirements.txt: Dependencies and libraries used

📈 Results

The CNN achieved high accuracy on the test data

The model can distinguish between benign and malignant classes

🖉 Future Work

Extend to multi-class classification

Test with other histopathological datasets

Deploy as a web application for clinical use

📣 Contact

Dipamoni Nath: [ndipamoni1@gmail.com] LinkedIn: linkedin.com/in/dipamoni-nath-568547247
