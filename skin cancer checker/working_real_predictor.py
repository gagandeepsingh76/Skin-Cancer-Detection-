import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def load_models():
    """Load all trained models"""
    print("Loading trained models...")
    print("="*60)
    
    model_files = {
        'CNN': 'cnn.keras',
        'MobileNet': 'mobilenet.keras',
        'EfficientNet': 'efficientnet.keras',
        'Inception': 'inception.keras',
        'DenseNet': 'desnet.keras'
    }
    
    models = {}
    existing_models = {}
    
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            existing_models[name] = file_size
            print(f"✓ {name}: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ {name}: {file_path} - NOT FOUND")
    
    if not existing_models:
        print("No models found!")
        return None, None
    
    print(f"\nFound {len(existing_models)} models successfully!")
    
    # Try to load models with TensorFlow
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        print("Loading models with TensorFlow...")
        
        for name, file_path in model_files.items():
            if name in existing_models:
                try:
                    print(f"Loading {name}...")
                    # Try different loading methods for compatibility
                    try:
                        # Method 1: Standard loading
                        model = tf.keras.models.load_model(file_path, compile=False)
                        print(f"✓ {name} loaded successfully (standard method)")
                    except:
                        try:
                            # Method 2: With custom_objects
                            model = tf.keras.models.load_model(file_path, compile=False, custom_objects={})
                            print(f"✓ {name} loaded successfully (custom_objects method)")
                        except:
                            # Method 3: Try loading as SavedModel
                            model = tf.keras.models.load_model(file_path, compile=False, custom_objects={}, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
                            print(f"✓ {name} loaded successfully (SavedModel method)")
                    
                    models[name] = model
                    
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
        
        if not models:
            print("No models could be loaded with TensorFlow")
            return None, existing_models
        
        return models, existing_models
        
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
        return None, existing_models

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values (0-1)
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_cancer_with_models(models, image_path):
    """Make real predictions using loaded models"""
    if not models:
        print("No models available for prediction")
        return None
    
    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None
    
    print(f"\nAnalyzing image: {image_path}")
    print("Making predictions with trained models...")
    print("-" * 50)
    
    results = {}
    
    for name, model in models.items():
        try:
            # Make prediction
            prediction = model.predict(processed_img, verbose=0)
            
            print(f"Raw prediction for {name}: {prediction.shape} - {prediction}")
            
            # Handle different output formats
            if len(prediction.shape) == 2:
                # Binary classification (cancer vs no cancer)
                if prediction.shape[1] == 2:
                    # Two classes: [no_cancer, cancer]
                    cancer_prob = prediction[0][1]  # Probability of cancer
                else:
                    # Two classes: [cancer, no_cancer] or other format
                    cancer_prob = prediction[0][0]  # First class probability
            else:
                # Single output
                cancer_prob = prediction[0][0]
            
            # Convert to percentage
            cancer_percentage = float(cancer_prob) * 100
            
            # Ensure percentage is within valid range
            cancer_percentage = max(0, min(100, cancer_percentage))
            
            results[name] = cancer_percentage
            print(f"✓ {name}: {cancer_percentage:.1f}%")
            
        except Exception as e:
            print(f"✗ Prediction failed for {name}: {e}")
            results[name] = None
    
    return results

def calculate_combined_result(results):
    """Calculate combined result from all models"""
    if not results:
        return None
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return None
    
    # Calculate combined percentage (average of all models)
    total_percentage = sum(valid_results.values())
    combined_percentage = total_percentage / len(valid_results)
    
    # Calculate confidence based on model agreement
    percentages = list(valid_results.values())
    std_dev = np.std(percentages)
    confidence = max(0, 100 - std_dev)  # Higher agreement = higher confidence
    
    return {
        'combined_percentage': combined_percentage,
        'confidence': confidence,
        'model_count': len(valid_results),
        'individual_results': valid_results
    }

def display_results(image_path, combined_result):
    """Display results in one frame"""
    if combined_result is None:
        return
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title('Test Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display individual model results
    individual_results = combined_result['individual_results']
    model_names = list(individual_results.keys())
    cancer_percentages = list(individual_results.values())
    
    bars = ax2.bar(model_names, cancer_percentages, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, cancer_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Individual Model Results', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cancer Probability (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Color code individual results
    for i, percentage in enumerate(cancer_percentages):
        if percentage > 70:
            color = '#FF6B6B'  # Red for high probability
        elif percentage > 40:
            color = '#FFA500'  # Orange for medium probability
        else:
            color = '#4ECDC4'  # Green for low probability
        bars[i].set_color(color)
    
    # Display combined result
    combined_percentage = combined_result['combined_percentage']
    confidence = combined_result['confidence']
    
    # Create a single bar for combined result
    ax3.bar(['COMBINED RESULT'], [combined_percentage], 
            color='#9B59B6', width=0.6)
    
    # Add percentage label
    ax3.text(0, combined_percentage + 1, f'{combined_percentage:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    ax3.set_title('FINAL COMBINED RESULT', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Cancer Probability (%)', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Add confidence indicator
    ax3.text(0, 10, f'Confidence: {confidence:.1f}%', 
             ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Print final result
    print("\n" + "="*60)
    print("FINAL COMBINED RESULT")
    print("="*60)
    print(f"COMBINED CANCER PROBABILITY: {combined_percentage:.1f}%")
    print(f"CONFIDENCE LEVEL: {confidence:.1f}%")
    print(f"MODELS USED: {combined_result['model_count']}")
    print("="*60)
    
    # Risk assessment
    if combined_percentage > 70:
        risk_level = "HIGH RISK"
        recommendation = "Immediate medical consultation recommended"
    elif combined_percentage > 40:
        risk_level = "MEDIUM RISK"
        recommendation = "Medical evaluation recommended within 1-2 weeks"
    else:
        risk_level = "LOW RISK"
        recommendation = "Regular monitoring recommended"
    
    print(f"RISK LEVEL: {risk_level}")
    print(f"RECOMMENDATION: {recommendation}")
    print("="*60)
    
    # Individual model breakdown
    print("\nINDIVIDUAL MODEL RESULTS:")
    for model_name, percentage in individual_results.items():
        if percentage > 70:
            status = "HIGH"
        elif percentage > 40:
            status = "MEDIUM"
        else:
            status = "LOW"
        print(f"{model_name:15} : {percentage:6.1f}% - {status} RISK")

def main():
    """Main function"""
    print("REAL SKIN CANCER PREDICTION TOOL")
    print("="*60)
    
    # Load models
    models, existing_models = load_models()
    
    if not existing_models:
        print("No model files found. Cannot proceed.")
        return
    
    # Check test image
    test_image = "1.png"
    if not os.path.exists(test_image):
        print(f"Test image {test_image} not found!")
        return
    
    if models:
        # Use real models for prediction
        print("\n" + "="*60)
        print("USING REAL TRAINED MODELS")
        print("="*60)
        results = predict_cancer_with_models(models, test_image)
    else:
        print("\n" + "="*60)
        print("MODELS COULD NOT BE LOADED")
        print("="*60)
        print("Please check TensorFlow installation and model files.")
        print("The models need to be loaded to make real predictions.")
        print("\nTroubleshooting:")
        print("1. Try: pip install tensorflow==2.13.0")
        print("2. Check if models are in correct format")
        print("3. Ensure Python version compatibility")
        return
    
    if results is None:
        print("Failed to get predictions. Cannot proceed.")
        return
    
    # Calculate combined result
    combined_result = calculate_combined_result(results)
    
    if combined_result is None:
        print("Failed to calculate combined result.")
        return
    
    # Display results
    display_results(test_image, combined_result)

if __name__ == "__main__":
    main()
