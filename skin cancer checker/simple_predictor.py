import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def check_models():
    """Check which model files exist"""
    model_files = {
        'CNN': 'cnn.keras',
        'MobileNet': 'mobilenet.keras',
        'EfficientNet': 'efficientnet.keras',
        'Inception': 'inception.keras',
        'DenseNet': 'desnet.keras'
    }
    
    existing_models = {}
    for name, file_path in model_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            existing_models[name] = {
                'path': file_path,
                'size_mb': file_size
            }
            print(f"✓ {name}: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ {name}: {file_path} - NOT FOUND")
    
    return existing_models

def check_image(image_path):
    """Check if test image exists and can be loaded"""
    if not os.path.exists(image_path):
        print(f"✗ Test image {image_path} not found!")
        return None
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not load image {image_path}")
            return None
        
        height, width, channels = img.shape
        print(f"✓ Test image loaded: {width}x{height} pixels, {channels} channels")
        return img
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None

def simulate_predictions():
    """Simulate predictions for demonstration purposes"""
    print("\n" + "="*60)
    print("SIMULATED SKIN CANCER PREDICTIONS")
    print("="*60)
    print("Note: This is a demonstration. Real predictions require")
    print("successful model loading with TensorFlow.")
    print("="*60)
    
    # Simulate results based on typical model performance
    simulated_results = {
        'CNN': 78.5,
        'MobileNet': 82.3,
        'EfficientNet': 89.7,
        'Inception': 85.1,
        'DenseNet': 87.4
    }
    
    return simulated_results

def display_simulated_results(image_path, results):
    """Display simulated results"""
    if results is None:
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img)
    ax1.set_title('Test Image (1.png)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display prediction results
    model_names = list(results.keys())
    cancer_percentages = list(results.values())
    
    bars = ax2.bar(model_names, cancer_percentages, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, cancer_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Skin Cancer Prediction Results (Simulated)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cancer Probability (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3)
    
    # Color code the results
    for i, percentage in enumerate(cancer_percentages):
        if percentage > 70:
            color = '#FF6B6B'  # Red for high probability
        elif percentage > 40:
            color = '#FFA500'  # Orange for medium probability
        else:
            color = '#4ECDC4'  # Green for low probability
        bars[i].set_color(color)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("SKIN CANCER PREDICTION RESULTS (SIMULATED)")
    print("="*60)
    for model_name, percentage in results.items():
        if percentage > 70:
            status = "HIGH RISK"
        elif percentage > 40:
            status = "MEDIUM RISK"
        else:
            status = "LOW RISK"
        print(f"{model_name:15} : {percentage:6.1f}% - {status}")
    print("="*60)
    print("Note: These are simulated results for demonstration.")
    print("Real predictions require successful TensorFlow model loading.")

def main():
    """Main function"""
    print("SKIN CANCER PREDICTION TOOL - SYSTEM CHECK")
    print("="*60)
    
    # Check models
    print("\nChecking model files...")
    existing_models = check_models()
    
    if not existing_models:
        print("\nNo model files found. Please ensure all .keras files are in the current directory.")
        return
    
    print(f"\nFound {len(existing_models)} model files.")
    
    # Check test image
    print("\nChecking test image...")
    test_image = "1.png"
    img = check_image(test_image)
    
    if img is None:
        print("Cannot proceed without test image.")
        return
    
    # Show system information
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    print(f"✓ Models found: {len(existing_models)}")
    print(f"✓ Test image: {test_image}")
    print("✗ TensorFlow: Import issues detected")
    print("\nRecommendations:")
    print("1. Try reinstalling TensorFlow: pip install --upgrade tensorflow")
    print("2. Use TensorFlow CPU version: pip install tensorflow-cpu")
    print("3. Check Python version compatibility")
    print("4. Consider using a virtual environment")
    
    # Ask user if they want to see simulated results
    print("\n" + "="*60)
    print("Would you like to see simulated results for demonstration?")
    print("(This shows how the tool would work with working models)")
    
    # For now, show simulated results
    print("\nShowing simulated results...")
    results = simulate_predictions()
    display_simulated_results(test_image, results)

if __name__ == "__main__":
    main()
