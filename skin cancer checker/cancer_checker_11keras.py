import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class SkinCancerChecker:
    def __init__(self):
        self.model_path = '11.keras'
        self.image_path = '1.png'
        self.model = None
        
    def load_model(self):
        """Load the 11.keras model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file {self.model_path} not found!")
                return False
                
            print(f"🔄 Loading model: {self.model_path}")
            # Suppress verbose output during model loading
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import warnings
            warnings.filterwarnings('ignore')
            
            self.model = keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Preprocess the image for the model"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image file {image_path} not found!")
                return None
                
            print(f"🔄 Loading and preprocessing image: {image_path}")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image {image_path}")
                return None
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224 (common input size for many models)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # Normalize pixel values to 0-1 range
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            print("✅ Image preprocessed successfully!")
            print(f"   - Original size: {img.shape}")
            print(f"   - Resized to: {img_resized.shape}")
            print(f"   - Normalized range: {img_normalized.min():.3f} - {img_normalized.max():.3f}")
            
            return img_batch, img_rgb
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict_cancer(self, img_batch):
        """Make prediction using the loaded model"""
        try:
            if self.model is None:
                print("❌ Model not loaded!")
                return None
                
            print("🔄 Making prediction...")
            
            # Get prediction
            prediction = self.model.predict(img_batch, verbose=0)
            
            print(f"✅ Prediction completed!")
            print(f"   - Raw prediction shape: {prediction.shape}")
            print(f"   - Raw prediction: {prediction}")
            
            # The model has 8 classes, we need to interpret them
            # Assuming class 0 is benign and classes 1-7 are different types of cancer
            # We'll calculate the probability of cancer as 1 - probability of class 0 (benign)
            if prediction.ndim > 1 and prediction.shape[1] == 8:
                # Get the probability of benign (class 0)
                benign_probability = float(prediction[0][0])
                # Cancer probability is 1 - benign probability
                cancer_probability = 1.0 - benign_probability
                
                # Also get the most likely class
                predicted_class = int(np.argmax(prediction[0]))
                class_probabilities = prediction[0]
                
                print(f"   - Class probabilities: {class_probabilities}")
                print(f"   - Predicted class: {predicted_class}")
                print(f"   - Benign probability: {benign_probability:.4f}")
                print(f"   - Cancer probability: {cancer_probability:.4f}")
                
                return cancer_probability, predicted_class, class_probabilities
            else:
                # Fallback for other output formats
                cancer_probability = float(prediction[0][0]) if prediction.shape[1] > 1 else float(prediction[0])
                cancer_probability = max(0.0, min(1.0, cancer_probability))
                return cancer_probability, 0, prediction[0]
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def interpret_result(self, probability, predicted_class=0, class_probabilities=None):
        """Interpret the prediction result"""
        print("\n" + "="*50)
        print("🔬 SKIN CANCER ANALYSIS RESULT")
        print("="*50)
        
        # Convert to percentage
        percentage = probability * 100
        
        print(f"📊 CANCER PROBABILITY: {percentage:.2f}%")
        
        # Show class information if available
        if class_probabilities is not None:
            print(f"🎯 PREDICTED CLASS: {predicted_class}")
            print("📋 CLASS PROBABILITIES:")
            class_names = ["Benign", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                          "Actinic Keratosis", "Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"]
            for i, (name, prob) in enumerate(zip(class_names, class_probabilities)):
                print(f"   {i}: {name}: {prob*100:.2f}%")
        
        # Determine risk level
        if percentage >= 70:
            risk_level = "🔴 HIGH RISK"
            recommendation = "Immediate medical consultation recommended"
        elif percentage >= 40:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "Medical evaluation recommended within 1-2 weeks"
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "Regular monitoring recommended"
        
        print(f"⚠️  RISK LEVEL: {risk_level}")
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        # Final verdict
        if percentage >= 50:
            verdict = "LIKELY TO HAVE CANCER"
            verdict_emoji = "⚠️"
        else:
            verdict = "LIKELY BENIGN (NO CANCER)"
            verdict_emoji = "✅"
        
        print(f"{verdict_emoji} VERDICT: {verdict}")
        print("="*50)
        
        return {
            'probability': percentage,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'verdict': verdict,
            'predicted_class': predicted_class,
            'class_probabilities': class_probabilities
        }
    
    def visualize_result(self, original_img, result):
        """Create a visualization of the result"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax1.imshow(original_img)
            ax1.set_title('Original Image (1.png)', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Result visualization
            probability = result['probability']
            colors = ['#2ECC71', '#F39C12', '#E74C3C']  # Green, Orange, Red
            if probability < 40:
                color = colors[0]
            elif probability < 70:
                color = colors[1]
            else:
                color = colors[2]
            
            ax2.bar(['Cancer Probability'], [probability], color=color, width=0.6)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Probability (%)', fontsize=12)
            ax2.set_title(f'Prediction Result: {probability:.2f}%', fontsize=14, fontweight='bold')
            
            # Add percentage text on bar
            ax2.text(0, probability + 2, f'{probability:.2f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=16)
            
            # Add risk level text
            ax2.text(0, -10, result['risk_level'], 
                    ha='center', va='top', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")
    
    def run_analysis(self):
        """Run the complete cancer analysis"""
        print("🚀 Starting Skin Cancer Analysis using 11.keras model")
        print("="*60)
        
        # Step 1: Load model
        if not self.load_model():
            return
        
        # Step 2: Preprocess image
        preprocessed = self.preprocess_image(self.image_path)
        if preprocessed is None:
            return
        
        img_batch, original_img = preprocessed
        
        # Step 3: Make prediction
        prediction_result = self.predict_cancer(img_batch)
        if prediction_result is None:
            return
        
        if isinstance(prediction_result, tuple):
            probability, predicted_class, class_probabilities = prediction_result
        else:
            probability = prediction_result
            predicted_class = 0
            class_probabilities = None
        
        # Step 4: Interpret result
        result = self.interpret_result(probability, predicted_class, class_probabilities)
        
        # Step 5: Visualize result
        self.visualize_result(original_img, result)
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Model used: {self.model_path}")
        print(f"🖼️  Image analyzed: {self.image_path}")

def main():
    """Main function to run the cancer checker"""
    checker = SkinCancerChecker()
    checker.run_analysis()

if __name__ == "__main__":
    main()
