import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import warnings

# Suppress all warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def check_cancer():
    """Simple function to check if image 1.png has cancer using 11.keras model"""
    
    print("SKIN CANCER CHECKER - Using 11.keras Model")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists('11.keras'):
        print("ERROR: Model file '11.keras' not found!")
        return
    
    if not os.path.exists('1.png'):
        print("ERROR: Image file '1.png' not found!")
        return
    
    try:
        # Load model
        print("Loading model...")
        model = keras.models.load_model('11.keras')
        print("Model loaded successfully!")
        
        # Load and preprocess image
        print("Loading image...")
        img = cv2.imread('1.png')
        if img is None:
            print("ERROR: Could not load image!")
            return
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print("Image preprocessed successfully!")
        print(f"   - Image size: {img.shape}")
        print(f"   - Resized to: {img_resized.shape}")
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_batch, verbose=0)
        
        print("Prediction completed!")
        print(f"   - Prediction shape: {prediction.shape}")
        print(f"   - Raw prediction: {prediction}")
        
        # Interpret results
        if prediction.shape[1] == 8:
            # 8-class classification
            class_names = ["Benign", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                          "Actinic Keratosis", "Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"]
            
            # Get probabilities
            probabilities = prediction[0]
            predicted_class = int(np.argmax(probabilities))
            max_probability = float(np.max(probabilities))
            
            # Calculate cancer probability (1 - benign probability)
            benign_probability = float(probabilities[0])
            cancer_probability = 1.0 - benign_probability
            
            print("\n" + "=" * 50)
            print("PREDICTION RESULTS")
            print("=" * 50)
            print(f"PREDICTED CLASS: {predicted_class} - {class_names[predicted_class]}")
            print(f"CONFIDENCE: {max_probability*100:.2f}%")
            print(f"CANCER PROBABILITY: {cancer_probability*100:.2f}%")
            print(f"BENIGN PROBABILITY: {benign_probability*100:.2f}%")
            
            print("\nALL CLASS PROBABILITIES:")
            for i, (name, prob) in enumerate(zip(class_names, probabilities)):
                print(f"   {i}: {name}: {prob*100:.2f}%")
            
            # Risk assessment
            print("\n" + "=" * 50)
            print("RISK ASSESSMENT")
            print("=" * 50)
            
            if cancer_probability >= 0.7:
                risk_level = "HIGH RISK"
                recommendation = "Immediate medical consultation recommended"
            elif cancer_probability >= 0.4:
                risk_level = "MEDIUM RISK"
                recommendation = "Medical evaluation recommended within 1-2 weeks"
            else:
                risk_level = "LOW RISK"
                recommendation = "Regular monitoring recommended"
            
            print(f"RISK LEVEL: {risk_level}")
            print(f"RECOMMENDATION: {recommendation}")
            
            # Final verdict
            if cancer_probability >= 0.5:
                verdict = "LIKELY TO HAVE CANCER"
            else:
                verdict = "LIKELY BENIGN (NO CANCER)"
            
            print(f"VERDICT: {verdict}")
            print("=" * 50)
            
        else:
            # Fallback for other output formats
            cancer_probability = float(prediction[0][0]) if prediction.shape[1] > 1 else float(prediction[0])
            cancer_probability = max(0.0, min(1.0, cancer_probability))
            
            print(f"\nCANCER PROBABILITY: {cancer_probability*100:.2f}%")
            
            if cancer_probability >= 0.5:
                print("LIKELY TO HAVE CANCER")
            else:
                print("LIKELY BENIGN (NO CANCER)")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    check_cancer()
