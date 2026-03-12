#  Food Classification Using CNN (34 Classes)

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Keras](https://img.shields.io/badge/Keras-CNN-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Deep Learning system for multi-class food image classification across **34 food categories**, implemented using CNN and transfer learning architectures, and deployed using a Flask web application.

---

##  Project Overview

This project implements a complete deep learning pipeline that:

- Performs multi-class image classification (34 classes)
- Compares three different model architectures
- Generates detailed evaluation reports
- Deploys a real-time prediction web interface
- Integrates nutritional information using a structured JSON dataset

The system demonstrates production-ready ML workflow from data preprocessing to deployment.

---

##  Objective

To design and deploy a scalable food image classification system that:

1. Classifies uploaded images into one of 34 food categories  
2. Allows dynamic selection between:
   - Custom CNN  
   - VGG16 (Transfer Learning)  
   - ResNet (Transfer Learning)  
3. Displays:
   - Predicted class
   - Confidence score
   - Evaluation metrics
   - Nutritional details for the predicted food item  

---

##  Dataset

- **Total Classes:** 34  
- **Balanced Images per Class:** 200  
- **Total Images Used:** 6,800  
- **Training Platform:** Kaggle (GPU)  

The dataset was manually balanced to ensure equal representation across all classes and prevent model bias.

The dataset is not included in this repository due to GitHub file size limitations.

Dataset size exceeds 100MB limit.

Download Dataset: Kaggle Dataset Link: https://www.kaggle.com/datasets/hemamalini33/food-classification-34-classes

After downloading, place the dataset inside:
```
static/food34_200_per_class/ 
                           ├── train/
                           ├── val/
                           └── test/
```
---

##  Data Preprocessing

### 1️. Dataset Balancing
- Selected 200 images per class  
- Eliminated class imbalance issues  

### 2️. Image Processing
- Resized images to 224×224  
- Normalized pixel values  
- Applied optional data augmentation:
  - Rotation  
  - Horizontal flip  
  - Zoom  

### 3️. Training Architecture
- Object-oriented implementation  
- Modular model training classes  
- Exception handling for robustness  

---

##  Model Architectures

###  1. Custom CNN (Baseline)
- Built from scratch
- Multiple Conv → Pool → Dense layers
- Trained for 30 epochs
- Saved as `custom_model.h5`

---

###  2. VGG16 (Transfer Learning)
- Pretrained on ImageNet
- Feature extractor layers frozen initially
- Custom classifier head added
- Trained for 50 epochs
- Saved as `vgg16_model.h5`

---

###  3. ResNet (Transfer Learning)
- Pretrained on ImageNet
- Residual architecture for deeper learning
- Custom classifier head added
- Trained for 50 epochs
- Saved as `resnet_model.h5`

---

##  Model Evaluation

Each model was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- TP / TN / FP / FN  

Evaluation reports are saved as:

```
Custom_Model.txt
VGG16_Model.txt
ResNet_Model.txt
```


These reports allow structured performance comparison across architectures.

---

##  Nutritional Information Integration

A JSON file stores nutritional information for each of the 34 food categories:

- Protein  
- Fiber  
- Calories  
- Carbohydrates  
- Fat  

The Flask application dynamically retrieves and displays this data after prediction.

---

##  Deployment (Flask Web Application)

The trained models are deployed through a Flask-based web interface.

### Features:
- Image upload functionality  
- Model selection dropdown  
- Real-time prediction  
- Confidence score display  
- Nutritional information output
- The web application allows users to select and run any of the three trained models (Custom CNN, VGG16, or ResNet50) to perform food image classification and view the predicted class along with nutritional information.  

---

##  Run Locally

```bash
pip install -r requirements.txt
python app.py
```
**Open in browser:**
```
http://localhost:5000
```

**Tech Stack**

- Python
- TensorFlow / Keras
- Flask
- NumPy
- OpenCV
- HTML / CSS / JavaScript

**Project Structure**

```
Food_Classification_Using_CNN/
│
├── app.py
├── requirements.txt
│
├── Data_Augmentation/
│   ├── data_augmentation.log
│   └── data_augmentation.pkl
│
├── json_folder/
│   ├── apple_pie.json
│   ├── burger.json
│   ├── butter_naan.json
│   └── ... (34 food class JSON files)
│
├── models/ (models files - not included in repository due to GitHub file size limitations)
│   ├── custom_cnn_food_model.h5
│   ├── ResNet_Model.h5
│   ├── vgg16_food_model.h5
│   └── json_all_food_classes.pkl
│
├── notebooks/
│   └── all-model-training.ipynb
│
├── Reports/
│   ├── Custom_CNN_Model_Report.txt
│   ├── ResNet_Model.txt
│   └── VGG16_Model.txt
│
├── static/ (Dataset folder - not included in repository due to GitHub file size limitations)
│   └── food34_200_per_class/
│       ├── train/
│       │   └── [34 food class folders]
│       ├── val/
│       │   └── [34 food class folders]
│       └── test/
│           └── [34 food class folders]
│
└── templates/
    └── index.html
```
**Training Configuration**

- Epochs: 50
- Image Size: 224×224
- Validation Split: 10%
- GPU Used: Kaggle

**Future Improvements**

- Add more food categories
- Mobile app integration
- Real-time calorie estimation
- Deploy to cloud platform

**Note**
Trained model files (.h5) are excluded due to GitHub size limitations.

**Author**

Hema Malini Gangumalla

Aspiring Data Scientist

📧 hemamalinig07@gmail.com

**License**

MIT License
