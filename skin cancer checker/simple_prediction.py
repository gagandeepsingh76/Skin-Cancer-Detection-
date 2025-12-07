import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Suppress all warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def predict_cancer():
    try:
        # Load model
        print("Loading model...")
        model = keras.models.load_model("11.keras")
        print("Model loaded successfully!")
        
        # Load and preprocess image
        print("Processing image...")
        img = Image.open("1.png")
        img = img.convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array, verbose=0)
        
        # The model has 8 classes
        if prediction.ndim > 1 and prediction.shape[1] == 8:
            # Get the probability of benign (class 0)
            benign_probability = float(prediction[0][0])
            # Cancer probability is 1 - benign probability
            cancer_probability = 1.0 - benign_probability
            
            # Also get the most likely class
            predicted_class = int(np.argmax(prediction[0]))
            class_probabilities = prediction[0]
            
            print(f"\nClass probabilities: {class_probabilities}")
            print(f"Predicted class: {predicted_class}")
            print(f"Benign probability: {benign_probability:.4f}")
            print(f"Cancer probability: {cancer_probability:.4f}")
            
            # Class names
            class_names = ["Benign", "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                          "Actinic Keratosis", "Seborrheic Keratosis", "Dermatofibroma", "Vascular Lesion"]
            
            print(f"\nPredicted class name: {class_names[predicted_class]}")
            
            # Final result
            if cancer_probability >= 0.5:
                print(f"\nRESULT: LIKELY TO HAVE CANCER ({cancer_probability*100:.2f}%)")
            else:
                print(f"\nRESULT: LIKELY BENIGN - NO CANCER ({cancer_probability*100:.2f}%)")
                
            return True
        else:
            print("Unexpected prediction format")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    predict_cancer()
