**Breast Cancer Subtype Classification using CNN.**

This repository presents a deep learning approach for classifying breast cancer subtypes (benign vs malignant) from histopathological images using Convolutional Neural Networks (CNN). The project showcases practical applications of AI and machine learning in medical image analysis, leveraging the BreakHis dataset and built entirely on Google Colab.

**🔬 Project Overview**

**Title**: Cancer Subtype Identification from Histopathological Images using CNN
**Objective**: Automate the classification of breast cancer subtypes using a CNN model trained on histopathology images.
**Dataset**: Breast cancer histopathology images (sourced via Kaggle)
**Platform**: Google Colab

**🛠️ Tools & Libraries**

Python 3.x
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn

**📊 Methodology**

**Data Splitting**: Divided the dataset into training and test sets.
**Preprocessing**: Resized, normalized, and augmented images for improved model robustness.
**Model Architecture**: Designed a custom CNN using Keras' Sequential API.
**Training**: Used categorical crossentropy loss and the Adam optimizer.
**Evaluation**: Assessed performance using accuracy score, confusion matrix, ROC curve, and classification report.

**🔄 Workflow Summary**

**Data Acquisition**: Downloaded via Kaggle API (kaggle.json)
**Preprocessing**: Applied resizing, normalization, and augmentation
**Modeling**: CNN layers included Conv2D, MaxPooling2D, Dropout, Flatten, and Dense
**Visualization**: Trained model visualized with loss/accuracy graphs, ROC curve, and confusion matrix

**📁 Repository Contents**

**breast_cancer_histopathology.ipynb**: Main Jupyter Notebook with full pipeline
**requirements.txt**: List of required Python libraries
**confusion_matrix.png, roc_curve.png**: Model performance plots
**README.md**: Project documentation

**📈 Results**

The model achieved high accuracy on the validation/test set.
Successfully distinguished between benign and malignant breast cancer subtypes.
Demonstrated reliability through ROC curve and confusion matrix visualization

**🚀 Future Enhancements**

Extend to** multi-class classification** (invasive ductal carcinoma, lobular, etc.)
Experiment with **transfer learning** and pretrained models (e.g., VGG16, ResNet50
Deploy as a **web-based diagnostic tool** for clinical applications

**📫 Contact**

Email: ndipamoni1@gmail.com
LinkedIn: https://www.linkedin.com/in/dipamoni-nath-568547247


