from flask import Flask, jsonify, request
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from io import BytesIO
import time
import os

app = Flask(__name__)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model only once when the app starts
try:
    # Load Processor and Model once
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    # Apply Optimization
    if device == "cuda":
        model.half()  # Convert to FP16 for GPU acceleration
    elif device == "cpu":
        # Simple optimization for CPU, full quantization might not work for all models
        model = model.to(torch.device("cpu"))

    model.to(device).eval()  # Move to GPU/CPU & set eval mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    
if device == "cuda":
    model = torch.compile(model) 
@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    try:
        print("Request received")
        
        # Check if we're using the default local image or one from the request
        try:
            # First try to use local image path (make sure path is correct)
            image_path = "./DETR/image2.jpg"
            if os.path.exists(image_path):
                print(f"Opening local image: {image_path}")
                image = Image.open(image_path)
            else:
                print(f"Local image not found at: {image_path}")
                # Provide a fallback message
                return jsonify({'error': f'Image not found at {image_path}'}), 404
        except Exception as img_error:
            print(f"Error opening image: {str(img_error)}")
            return jsonify({'error': f'Error opening image: {str(img_error)}'}), 500

        print("Image loaded, processing...")
        
        # Process image with timeout handling
        start_time = time.time()
        
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Resize the image to match model's expected input
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Extract detected objects
        Objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            Objects.append({
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "box": box
            })
        
        process_time = time.time() - start_time
        print(f"Processing completed in {process_time:.2f} seconds")
        print(f"Detected objects: {Objects}")

        return jsonify({
            "objects": Objects,
            "processing_time": process_time
        })

    except Exception as e:
        # Handle any errors and log the exception message
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'device': device})

if __name__ == '__main__':
    # Set threaded=False for better performance with ML models
    # Higher timeout for slow model inference
    app.run(debug=False, threaded=False, host='0.0.0.0', port=5000)