from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import requests
import json
from groq import Groq

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['MODEL_PATH'] = 'trained_model.keras'  # Path to your existing model
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disease classes from your model
DISEASE_CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy']

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to load the model
def get_model():
    try:
        model = load_model(app.config['MODEL_PATH'])
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Adjust size according to your model requirements
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,0)
    return img_array

# Function to get disease information from Groq/Llama4
def get_disease_info_from_groq(disease_name):
    # Replace with your Groq API key and endpoint
    groq_api_key = "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" 
    groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Remove underscores and format disease name
    formatted_disease = disease_name.replace('___', ' - ').replace('_', ' ')
    
    # Create the prompt for the LLM
    prompt = f"""
    Please provide detailed information about the plant disease "{formatted_disease}" including:
    1. A brief description of the disease, its cause, and visual symptoms
    2. The pathogen type (fungus, bacteria, virus, etc.)
    3. Environmental conditions that favor the disease
    4. A detailed treatment plan
    5. Preventative measures for future cultivation

    The JSON must include the following fields:
    - description: A brief explanation of the disease, its causes, and visible symptoms.
    - pathogen_type: One of [fungus, bacteria, virus, pest, abiotic, unknown].
    - favorable_conditions: A list of environmental conditions that help the disease spread.
    - treatment: A list of recommended treatments.
    - prevention: A list of preventive measures to avoid the disease in future crops.

    Please return ONLY a JSON object. Do not include any extra text, notes, or formatting outside of the JSON.

    """
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_api_key}"
        }
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 1024
        }
        
        response = requests.post(groq_api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the generated text from the response
            content = result["choices"][0]["message"]["content"]
            # Parse the JSON string from the content
            try:
                disease_info = json.loads(content)
                return disease_info
            except json.JSONDecodeError:
                # If LLM doesn't return valid JSON, create a fallback response
                print("Error parsing LLM response as JSON")
                return {
                    "description": "Information could not be retrieved. Please try again.",
                    "pathogen_type": "Unknown",
                    "favorable_conditions": ["Information not available"],
                    "treatment": ["Information not available"],
                    "prevention": ["Information not available"]
                }
        else:
            print(f"Groq API Error: {response.status_code}")
            return get_fallback_info(disease_name)
            
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return get_fallback_info(disease_name)

# Fallback information when Groq API fails
def get_fallback_info(disease_name):
    # Basic information for common diseases
    if "healthy" in disease_name:
        return {
            "description": "The plant appears healthy with no signs of disease.",
            "pathogen_type": "None",
            "favorable_conditions": ["Normal growing conditions"],
            "treatment": ["No treatment needed", "Continue regular care"],
            "prevention": ["Regular monitoring", "Proper watering and fertilization"]
        }
    else:
        return {
            "description": f"Information for {disease_name} could not be retrieved.",
            "pathogen_type": "Unknown",
            "favorable_conditions": ["Information not available"],
            "treatment": ["Remove affected leaves", "Apply appropriate fungicide/pesticide", "Ensure proper plant spacing for airflow"],
            "prevention": ["Crop rotation", "Proper sanitation", "Use disease-resistant varieties when available"]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load model if not already loaded
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model loading failed'}), 500
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(file_path)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            trueness = float(predictions[0][predicted_class_index]) * 100
            
            # Get predicted class name
            predicted_class = DISEASE_CLASSES[predicted_class_index]
            
            # Get disease information from Groq/Llama3
            disease_info = get_disease_info_from_groq(predicted_class)
            
            # Prepare result
            result = {
                'disease_name': predicted_class.replace('___', ' - ').replace('_', ' '),
                'trueness': round(trueness, 2),
                'description': disease_info['description'],
                'pathogen_type': disease_info['pathogen_type'],
                'favorable_conditions': disease_info['favorable_conditions'],
                'treatment': disease_info['treatment'],
                'prevention': disease_info['prevention'],
                'image_path': f"/static/uploads/{filename}"
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)