import os
import sys
import warnings
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Suppress all warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class QuietCancerChecker:
    def __init__(self, model_path="11.keras", image_path="1.png"):
        self.model_path = model_path
        self.image_path = image_path
        self.model = None

    def load_model(self):
        """Load the 11.keras model quietly"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file {self.model_path} not found!")
                return False
                
            print("Loading model...")
            
            # Redirect stdout to suppress model loading output
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                self.model = keras.models.load_model(self.model_path)
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_image(self):
        """Preprocess the image for the model"""
        try:
            if not os.path.exists(self.image_path):
                print(f"Image file {self.image_path} not found!")
                return None
                
            print("Processing image...")
            
            # Load and preprocess image
            img = Image.open(self.image_path)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            print("Image processed successfully!")
            return img_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict_cancer(self, img_array):
        """Make prediction using the model"""
        try:
            print("Making prediction...")
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            
            # The model has 8 classes, we need to interpret them
            # Assuming class 0 is benign and classes 1-7 are different types of cancer
            if prediction.ndim > 1 and prediction.shape[1] == 8:
                # Get the probability of benign (class 0)
                benign_probability = float(prediction[0][0])
                # Cancer probability is 1 - benign probability
                cancer_probability = 1.0 - benign_probability
                
                # Also get the most likely class
                predicted_class = int(np.argmax(prediction[0]))
                class_probabilities = prediction[0]
                
                print(f"Class probabilities: {class_probabilities}")
                print(f"Predicted class: {predicted_class}")
                print(f"Benign probability: {benign_probability:.4f}")
                print(f"Cancer probability: {cancer_probability:.4f}")
                
                return cancer_probability, predicted_class, class_probabilities
            else:
                print("Unexpected prediction format")
                return None
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def interpret_result(self, probability, predicted_class=0, class_probabilities=None):
        """Interpret the prediction result"""
        print("\n" + "="*50)
        print("SKIN CANCER ANALYSIS RESULT")
        print("="*50)
        
        # Convert to percentage
        percentage = probability * 100
        
        print(f"CANCER PROBABILITY: {percentage:.2f}%")
        
        # Show class information if available
        if class_probabilities is not None:
            print(f"PREDICTED CLASS: {predicted_class}")
            print("CLASS PROBABILITIES:")
            class_names = ["Benign", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                          "Actinic Keratosis", "Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"]
            for i, (name, prob) in enumerate(zip(class_names, class_probabilities)):
                print(f"   {i}: {name}: {prob*100:.2f}%")
        
        # Determine risk level
        if percentage >= 70:
            risk_level = "HIGH RISK"
            recommendation = "Immediate medical consultation recommended"
        elif percentage >= 40:
            risk_level = "MEDIUM RISK"
            recommendation = "Medical evaluation recommended within 1-2 weeks"
        else:
            risk_level = "LOW RISK"
            recommendation = "Regular monitoring recommended"
        
        print(f"RISK LEVEL: {risk_level}")
        print(f"RECOMMENDATION: {recommendation}")
        
        # Final verdict
        if percentage >= 50:
            verdict = "LIKELY TO HAVE CANCER"
        else:
            verdict = "LIKELY BENIGN (NO CANCER)"
        
        print(f"VERDICT: {verdict}")
        print("="*50)

    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting skin cancer analysis...")
        print(f"Model: {self.model_path}")
        print(f"Image: {self.image_path}")
        print()
        
        # Step 1: Load model
        if not self.load_model():
            return False
        
        # Step 2: Preprocess image
        img_array = self.preprocess_image()
        if img_array is None:
            return False
        
        # Step 3: Make prediction
        prediction_result = self.predict_cancer(img_array)
        if prediction_result is None:
            return False
        
        # Handle different return formats
        if isinstance(prediction_result, tuple):
            probability, predicted_class, class_probabilities = prediction_result
        else:
            probability = prediction_result
            predicted_class = 0
            class_probabilities = None
        
        # Step 4: Interpret result
        result = self.interpret_result(probability, predicted_class, class_probabilities)
        
        return True

def main():
    """Main function"""
    checker = QuietCancerChecker()
    success = checker.run_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed!")

if __name__ == "__main__":
    main()
