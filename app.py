import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===== PERFECT CONFIGURATION =====
class Config:
    MODELS_DIR = r'C:\Users\geeth\PycharmProjects\Food_Classification_CNN\models'
    JSON_FOLDER = r'C:\Users\geeth\PycharmProjects\Food_Classification_CNN\json_folder'

    # CORRECTED PATHS
    CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, 'custom_cnn_food_model.h5')
    VGG16_MODEL_PATH = os.path.join(MODELS_DIR, 'vgg16_food_model.h5')
    RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'ResNet_Model.h5')

    CUSTOM_REPORT = os.path.join(MODELS_DIR, 'Custom_CNN_Model_Report.txt')
    VGG16_REPORT = os.path.join(MODELS_DIR, 'VGG16_Model.txt')
    RESNET_REPORT = os.path.join(MODELS_DIR, 'ResNet_Model.txt')

    IMG_SIZE = (224, 224)

# ===== PERFECT CLASS NAMES =====
CLASS_NAMES = [
    "Baked_potato", "Crispy_chicken", "Donut", "Fries", "Hot_Dog", "Sandwich",
    "Taco", "Taquito", "apple_pie", "burger", "butter_naan", "chai", "chapati",
    "cheesecake", "chicken_curry", "chole_bhature", "dal_makhani", "dhokla",
    "fried_rice", "ice_cream", "idli", "jalebi", "kadai_paneer", "kathi_rolls",
    "kulfi", "masala_dosa", "momos", "omlette", "paani_puri", "pakode",
    "pav_bhaji", "pizza", "samosa", "sushi"
]

# Global variables
models = {}
nutrition_data = {}
model_metrics = {}
available_models = []  # Track which models are actually available

def load_models():
    """Load available models with error handling"""
    try:
        logger.info("Loading models for predictions...")
        loaded_count = 0
        global available_models
        available_models = []

        # Load Custom Model
        if os.path.exists(Config.CUSTOM_MODEL_PATH):
            try:
                models['custom'] = load_model(Config.CUSTOM_MODEL_PATH, compile=False)
                available_models.append('custom')
                logger.info("Custom model loaded")
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading custom model: {str(e)}")

        # Load VGG16 Model
        if os.path.exists(Config.VGG16_MODEL_PATH):
            try:
                models['vgg16'] = load_model(Config.VGG16_MODEL_PATH, compile=False)
                available_models.append('vgg16')
                logger.info("VGG16 model loaded")
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading VGG16 model: {str(e)}")

        # Load ResNet Model (optional - don't fail if missing)
        if os.path.exists(Config.RESNET_MODEL_PATH):
            try:
                models['resnet'] = load_model(Config.RESNET_MODEL_PATH, compile=False)
                available_models.append('resnet')
                logger.info("ResNet model loaded")
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading ResNet model: {str(e)}")
        else:
            logger.warning(f"ResNet model not found at: {Config.RESNET_MODEL_PATH}")
            logger.warning("Application will continue with available models")

        logger.info(f"Successfully loaded {loaded_count} out of 3 models")
        logger.info(f"Available models: {available_models}")
        return loaded_count > 0  # Return True if at least one model loaded

    except Exception as e:
        logger.error(f"Unexpected error loading models: {str(e)}")
        return False

def load_nutrition_data():
    """Load nutrition data from JSON files"""
    try:
        logger.info("Loading nutrition data...")
        loaded_count = 0

        if not os.path.exists(Config.JSON_FOLDER):
            logger.error(f"JSON folder not found: {Config.JSON_FOLDER}")
            # Create default nutrition data
            for class_name in CLASS_NAMES:
                nutrition_data[class_name] = {
                    'calories': 'N/A', 'protein': 'N/A', 'carbohydrates': 'N/A',
                    'fat': 'N/A', 'fiber': 'N/A'
                }
            return True  # Continue with default data

        for class_name in CLASS_NAMES:
            json_path = os.path.join(Config.JSON_FOLDER, f"{class_name}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        nutrition_data[class_name] = data.get('nutrition', {})
                    loaded_count += 1
                    logger.info(f"Loaded nutrition for: {class_name}")
                except Exception as e:
                    logger.error(f"Error loading {class_name}.json: {str(e)}")
                    nutrition_data[class_name] = {
                        'calories': 'N/A', 'protein': 'N/A', 'carbohydrates': 'N/A',
                        'fat': 'N/A', 'fiber': 'N/A'
                    }
            else:
                logger.warning(f"Nutrition file not found: {json_path}")
                nutrition_data[class_name] = {
                    'calories': 'N/A', 'protein': 'N/A', 'carbohydrates': 'N/A',
                    'fat': 'N/A', 'fiber': 'N/A'
                }

        logger.info(f"Nutrition data loaded for {loaded_count} out of {len(CLASS_NAMES)} classes")
        return True

    except Exception as e:
        logger.error(f"Error loading nutrition data: {str(e)}")
        # Create default nutrition data
        for class_name in CLASS_NAMES:
            nutrition_data[class_name] = {
                'calories': 'N/A', 'protein': 'N/A', 'carbohydrates': 'N/A',
                'fat': 'N/A', 'fiber': 'N/A'
            }
        return True  # Continue with default data

def parse_metrics_file(file_path):
    """Parse metrics from report files"""
    try:
        if not os.path.exists(file_path):
            return {"error": "Metrics file not found"}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metrics = {}
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if 'Accuracy:' in line:
                try:
                    metrics['accuracy'] = float(line.split(':')[1].strip())
                except:
                    metrics['accuracy'] = line.split(':')[1].strip()
            elif 'Precision:' in line:
                try:
                    metrics['precision'] = float(line.split(':')[1].strip())
                except:
                    metrics['precision'] = line.split(':')[1].strip()
            elif 'Recall:' in line:
                try:
                    metrics['recall'] = float(line.split(':')[1].strip())
                except:
                    metrics['recall'] = line.split(':')[1].strip()
            elif 'F1-Score:' in line:
                try:
                    metrics['f1_score'] = float(line.split(':')[1].strip())
                except:
                    metrics['f1_score'] = line.split(':')[1].strip()
            elif 'TP:' in line:
                metrics['tp'] = line.split(':')[1].strip()
            elif 'TN:' in line:
                metrics['tn'] = line.split(':')[1].strip()
            elif 'FP:' in line:
                metrics['fp'] = line.split(':')[1].strip()
            elif 'FN:' in line:
                metrics['fn'] = line.split(':')[1].strip()

        return metrics
    except Exception as e:
        return {"error": f"Error parsing metrics: {str(e)}"}

def load_model_metrics():
    """Load model performance metrics"""
    try:
        logger.info("Loading model metrics...")
        loaded_count = 0

        report_files = {
            'custom': Config.CUSTOM_REPORT,
            'vgg16': Config.VGG16_REPORT,
            'resnet': Config.RESNET_REPORT
        }

        for model_name, report_path in report_files.items():
            if os.path.exists(report_path):
                model_metrics[model_name] = parse_metrics_file(report_path)
                loaded_count += 1
                logger.info(f"Loaded metrics for {model_name}")
            else:
                model_metrics[model_name] = {"error": f"Metrics file not found: {report_path}"}
                logger.warning(f"Metrics file not found: {report_path}")

        logger.info(f"Loaded metrics for {loaded_count} out of 3 models")
        return True

    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        # Set default metrics
        for model_name in ['custom', 'vgg16', 'resnet']:
            model_metrics[model_name] = {"error": "Metrics not available"}
        return True

def preprocess_image_perfectly(img, target_size, model_type):
    """Preprocess image for prediction"""
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array

    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def predict_with_model(model, processed_image, model_name):
    """Make prediction with the trained model"""
    try:
        predictions = model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = CLASS_NAMES[predicted_idx]

        logger.info(f"{model_name} Prediction: {predicted_class} (Confidence: {confidence:.4f})")
        return predicted_class, confidence

    except Exception as e:
        raise Exception(f"Prediction failed for {model_name}: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/available_models')
def get_available_models():
    """Return list of available models to frontend"""
    return jsonify({'available_models': available_models})

@app.route('/predict', methods=['POST'])
def predict_perfectly():
    """Main prediction endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})

        image_file = request.files['image']
        model_type = request.form.get('model_type', 'custom').lower()

        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        if model_type not in models:
            available_models_list = list(models.keys())
            return jsonify({
                'success': False,
                'error': f'Model {model_type} not available. Available models: {", ".join(available_models_list)}'
            })

        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
        if not '.' in image_file.filename or \
                image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Use PNG, JPG, or JPEG'})

        img = Image.open(image_file.stream)
        processed_image = preprocess_image_perfectly(img, Config.IMG_SIZE, model_type)

        predicted_class, confidence = predict_with_model(
            models[model_type], processed_image, model_type.upper()
        )

        nutrition_info = nutrition_data.get(predicted_class, {
            'calories': 'N/A', 'protein': 'N/A', 'carbohydrates': 'N/A',
            'fat': 'N/A', 'fiber': 'N/A'
        })

        metrics_data = model_metrics.get(model_type, {"error": "Model metrics not available"})

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'nutrition_info': nutrition_info,
            'model_used': model_type.upper(),
            'model_metrics': metrics_data,
            'image_data': f"data:image/jpeg;base64,{img_str}"
        }

        logger.info(f"PREDICTION COMPLETED: {predicted_class} with {model_type.upper()}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'})

@app.route('/health')
def health_check():
    """Check if all components are loaded properly"""
    status = {
        'models_loaded': len(models),
        'available_models': available_models,
        'nutrition_data_loaded': len(nutrition_data) > 0,
        'metrics_loaded': len(model_metrics) > 0,
        'total_classes': len(CLASS_NAMES)
    }
    return jsonify(status)

# Initialize the application
def initialize_app():
    """Initialize all components with robust error handling"""
    logger.info("INITIALIZING FOOD CLASSIFICATION SYSTEM...")

    # Load models (at least one required)
    models_loaded = load_models()
    if not models_loaded:
        logger.error("CRITICAL: No models could be loaded!")
        return False

    # Load nutrition data (optional - can run without)
    nutrition_loaded = load_nutrition_data()

    # Load metrics (optional - can run without)
    metrics_loaded = load_model_metrics()

    logger.info("READY FOR PREDICTIONS!")
    logger.info(f"Available models: {available_models}")
    logger.info(f"Nutrition data: {'Loaded' if nutrition_loaded else 'Not available'}")
    logger.info(f"Model metrics: {'Loaded' if metrics_loaded else 'Not available'}")

    return True

if __name__ == '__main__':
    # Initialize the application
    if initialize_app():
        logger.info("Starting Flask server on http://localhost:5000")
        logger.info("Available models: " + ", ".join(available_models))
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Cannot start server - no models available")