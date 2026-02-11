# Food Classification Using CNN (34 Classes)

This project is a complete deep learning pipeline for food image classification across 34 food categories.
It includes:

- Dataset balancing and preprocessing
- Training **three models** (Custom CNN, VGG16, ResNet)
- Model evaluation with detailed metrics
- A **Flask web app** for image-based predictions
- A **JSON file** with nutritional information for each food class

#  Project Objective
Build and deploy a deep learning-based system that can:

1. Classify an input food image into one of **34 classes**.
2. Let the user choose which model to use: **Custom CNN**, **VGG16**, or **ResNet**.
3. Display:
     - The predicted class
     - Model evaluation metrics (Accuracy, TP, TN, FP, FN, Precision, Recall, F1-Score)
     - Nutritional information (protein, fiber, calories, carbs, fat) for the predicted class.

# Dataset
- **Total Classes:** 34
- **Images per Class (after balancing)**: 200
- **Total Images Used**: 34 × 200 = 6,800
- **Dataset Source**: Food Image Classification Dataset
  - (Replace this with your actual link, e.g. Kaggle / Google Drive / etc.)
     
# Preprocessing Steps
1. **Balancing the Dataset**
   - From the original dataset, **200 images per class** were selected.
   - This helps avoid class imbalance and gives each class equal representation.
     
2. **Image Transformations (Typical)**
   - Resizing images (e.g., 224×224 or 256×256 depending on the model).
   - Normalization (scaling pixel values).
   - Data augmentation (optional, e.g., rotation, flip, zoom) to improve generalization.
     
3. **Storage**
   - Model training and experimentation were performed on Kaggle (GPU)..
     
# Model Development
Three different models were implemented and trained for comparison:

1. **Custom Deep Learning Model (Custom CNN)**
2.**VGG16 (Transfer Learning)**
3. **ResNet (Transfer Learning)**
All models were trained using **object-oriented Python classes**, with **exception handling** for robustness.

# Common Training Setup
- **Epochs**: 50 (for each model)
- **Loss Function**: (e.g., Categorical Crossentropy for multi-class classification)
- **Optimizer**: (e.g., Adam / SGD – specify what you used)
- **Batch Size**: (fill in your value)
- **Framework**: TensorFlow / Keras / PyTorch (mention what you actually used)
  
Each model generates a validation report containing:

- Accuracy
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Precision
- Recall
- F1-Score
  
The reports are saved as:

- ```Custom_Model.txt```
- ```VGG16_Model.txt```
- ```ResNet_Model.txt```

# 1. Custom Deep Learning Model
This is a CNN model built from scratch without using pretrained weights.  

# Key Points
- Built using multiple convolutional, pooling, and dense layers.
- Suitable as a baseline model for comparison with transfer learning models.
- Trained for 30 epochs.
- The final trained model is saved (e.g., ```custom_model.h5``` or similar).
  
# Output
- Training & validation accuracy/loss.
- Saved evaluation metrics in ```Custom_Model.txt```.
  
# 2. VGG16 Model (Transfer Learning)
This model uses VGG16 pretrained on ImageNet as a feature extractor.

# Key Points
- Base VGG16 layers are usually **frozen initially**.
- A custom classifier head (Dense layers) is added on top for 34-class classification.
- Trained for **50 epochs**.
- The final trained model is saved (e.g., ```vgg16_model.h5```).
  
# Output
- Performance metrics saved in ```VGG16_Model.txt```.
- Typically achieves better accuracy than the custom CNN due to transfer learning.
  
# 3. ResNet Model (Transfer Learning)
This model uses ResNet (e.g., ResNet50, ResNet101) pre-trained on **ImageNet**.

# Key Points
- Uses residual connections to train very deep networks effectively.
- A custom classifier is added on top for 34 classes.
- Trained for **50 epochs**.
- The final trained model is saved (e.g., ```resnet_model.h5```).
  
# output
- Performance metrics saved in ```ResNet_Model.txt```.
- Often provides strong results and generalization.
  
#  Evaluation Metrics
For each model, the following metrics are computed on the validation/test set:

- **Accuracy**: Overall correctness of predictions.
- **TP (True Positives)**: Correctly predicted positive samples.
- **TN (True Negatives)**: Correctly predicted negative samples.
- **FP (False Positives)**: Incorrectly predicted positives.
- **FN (False Negatives)**: Missed positives.
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall.

All of these are summarized in text files:
- ```Custom_Model.txt```
- ```VGG16_Model.txt```
- ```ResNet_Model.txt```
  
You can open these files to compare which model performs best.
**Exammple**:- 

<img width="400" height="200" alt="Screenshot 2026-01-31 103814" src="https://github.com/user-attachments/assets/9fda08b5-2b01-4c76-9115-0e1038b0d8be" />

#  Nutritional Information (JSON File)
Each of the 34 food classes has associated nutritional data.

For every class, the JSON file stores:

- **Protein**
- **Fiber**
- **Calories**
- **Carbohydrates**
- **Fat**

## Tech Stack
- Python, TensorFlow/Keras
- Flask
- OpenCV, NumPy
- HTML, CSS, JavaScript

## Project Structure
Food_Classification_Using_CNN/
├── app.py

├── templates/

├── json_folder/

├── notebooks/

├── data_augmentation/

├── models/

├── requirements.txt

└── README.md

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Open browser:

```
http://localhost:5000
```

## Features
- Image upload and prediction
- Multiple model comparison
- Confidence score display
- Clean web interface

## Training Details
- Epochs: 50
- Image size: 224×224
- Validation split: 10%
- Platform: Kaggle (GPU)

## Future Work
- Increase food classes
- Mobile application
- Calorie estimation

## Note
- Trained model files (.h5 / .pkl) are not included due to size limitations.

## Author
**Hema Malini**
