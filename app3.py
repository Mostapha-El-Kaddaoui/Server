from flask import Flask, jsonify, request
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
from io import BytesIO  # Import BytesIO for image content handling
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

app = Flask(__name__)

#processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
#model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")



@app.route('/phonecall', methods=['POST'])
def PhoneCall():
    try:
        url = "https://raw.githubusercontent.com/Mostapha-El-Kaddaoui/Server-Flask-Railway/refs/heads/main/image2.jpg"
        image = Image.open(requests.get(url, stream=True).raw)  # Ensure to use BytesIO for image content
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Resize the image to match model's expected input
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Extract detected objects
        Objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            Objects.append(model.config.id2label[label.item()])
        

        # Check the request data
        data = request.json
        if data and data.get('message') == 'hello':
            return jsonify(Objects)  # Use jsonify instead of json.dumps
        else:
            return jsonify({'error': 'Invalid request'}), 400

    except Exception as e:
        # Handle any errors and log the exception message
        print(f"Error occurred: {str(e)}")  # Print the error for debugging
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
