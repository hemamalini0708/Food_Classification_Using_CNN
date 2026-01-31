# Food Classification Using CNN (34 Classes)

End-to-end deep learning project to classify 34 food items using CNN and transfer learning (VGG16, ResNet50), deployed with Flask.

<img width="932" height="538" alt="image" src="https://github.com/user-attachments/assets/f1d76032-904d-4f5b-a97f-f390f0f4cf1e" />


## Overview
This project identifies food items from images and displays the predicted class with confidence score.  
It compares a custom CNN with transfer learning models to select the best-performing approach.

## Dataset
- Source: Kaggle Food Image Classification Dataset  
- Classes: 34 food categories  
- Images resized to 224×224  
- Data augmentation applied

## Models Used
- Custom CNN (from scratch)
- VGG16 (transfer learning)
- ResNet50 (transfer learning)

<img width="927" height="415" alt="Screenshot 2026-01-31 103814" src="https://github.com/user-attachments/assets/9fda08b5-2b01-4c76-9115-0e1038b0d8be" />

<img width="932" height="439" alt="Screenshot 2026-01-31 103806" src="https://github.com/user-attachments/assets/678680d2-6a97-4309-bd70-eb4da1d5a2f6" />

<img width="927" height="455" alt="Screenshot 2026-01-31 103754" src="https://github.com/user-attachments/assets/e18cfa11-cfa8-4454-ab4d-e79207b94e63" />

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
