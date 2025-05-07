from flask import Flask, request, render_template, send_from_directory
import os
from datetime import datetime
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Directory to store images and detection results
UPLOAD_FOLDER = 'frames'
RESULTS_FOLDER = 'results'
COORDINATES_FILE = 'detected_coordinates.json'

# Create necessary directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLOv8 model
model = None

def load_model():
    global model
    try:
        model = YOLO('best.pt')
        print("YOLOv8 model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return False

def detect_weeds(image_path):
    """Detect weeds in the image and return coordinates"""
    if model is None:
        if not load_model():
            return []
    
    try:
        # Run YOLOv8 inference on the image
        results = model(image_path)
        detections = []
        
        # Process the results (first result if we're processing a single image)
        result = results[0]
        
        # Extract detection information
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                # Get box coordinates (in xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Get confidence score
                confidence = round(float(box.conf[0].item()), 2)
                
                # Simplified detection data format
                detections.append({
                    'c': confidence,
                    'x': center_x,
                    'y': center_y
                })
        
        return detections
    
    except Exception as e:
        print(f"Error during weed detection: {e}")
        return []

def save_coordinates(image_filename, detections):
    """Save detection coordinates to JSON file"""
    coordinates_path = os.path.join(app.config['RESULTS_FOLDER'], COORDINATES_FILE)
    
    # Load existing coordinates if file exists
    if os.path.exists(coordinates_path):
        with open(coordinates_path, 'r') as f:
            try:
                all_coordinates = json.load(f)
            except json.JSONDecodeError:
                all_coordinates = {}
    else:
        all_coordinates = {}
    
    # Add new detections
    all_coordinates[image_filename] = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'detections': detections
    }
    
    # Save updated coordinates
    with open(coordinates_path, 'w') as f:
        json.dump(all_coordinates, f, indent=2)

def draw_detections(image_path, detections):
    """Draw detection boxes on the image and save the result"""
    image = cv2.imread(image_path)
    
    for detection in detections:
        # Draw center point
        center_x = detection['x']
        center_y = detection['y']
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add confidence label
        label = f"c: {detection['c']:.2f}"
        cv2.putText(image, label, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the annotated image
    filename = os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"detected_{filename}")
    cv2.imwrite(result_path, image)
    
    return f"detected_{filename}"

@app.route('/')
def index():
    """Homepage showing the latest image and detections"""
    # Find the most recent image in the frames directory
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    latest_image = None
    detected_image = None
    detection_data = None
    
    if image_files:
        # Sort by modification time, newest first
        latest_image = sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)[0]
        
        # Check if we have a corresponding detected image
        detected_filename = f"detected_{latest_image}"
        detected_path = os.path.join(app.config['RESULTS_FOLDER'], detected_filename)
        
        if os.path.exists(detected_path):
            detected_image = detected_filename
        
        # Try to load detection data
        coordinates_path = os.path.join(app.config['RESULTS_FOLDER'], COORDINATES_FILE)
        if os.path.exists(coordinates_path):
            try:
                with open(coordinates_path, 'r') as f:
                    all_coordinates = json.load(f)
                if latest_image in all_coordinates:
                    detection_data = all_coordinates[latest_image]
            except Exception as e:
                print(f"Error loading detection data: {e}")
    
    return render_template('index.html', 
                          latest_image=latest_image, 
                          detected_image=detected_image,
                          detection_data=detection_data)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve detection result images"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint to receive uploaded images from Raspberry Pi"""
    if 'image' not in request.files:
        return 'No image part', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weed_scan_{timestamp}.jpg"
    
    # Save the file
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    
    # Run weed detection
    detections = detect_weeds(image_path)
    
    # Save detection coordinates
    save_coordinates(filename, detections)
    
    # Draw detections on the image
    if detections:
        draw_detections(image_path, detections)
    
    return 'Image uploaded and processed successfully', 200

@app.route('/coordinates')
def get_coordinates():
    """Endpoint to retrieve all detected coordinates"""
    coordinates_path = os.path.join(app.config['RESULTS_FOLDER'], COORDINATES_FILE)
    
    if os.path.exists(coordinates_path):
        return send_from_directory(app.config['RESULTS_FOLDER'], COORDINATES_FILE)
    else:
        return "No detections available yet", 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create a simple HTML template file
    template_path = os.path.join('templates', 'index.html')
    with open(template_path, 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Weed Detection Monitor</title>
    <meta http-equiv="refresh" content="10"> <!-- Auto-refresh every 10 seconds -->
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #4CAF50;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .image-container {
            flex: 1;
            min-width: 300px;
        }
        .data-container {
            flex: 1;
            min-width: 300px;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .no-image {
            padding: 20px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .detection-title {
            margin-top: 0;
            color: #4CAF50;
        }
    </style>
</head>
<body>
    <h1>Raspberry Pi Weed Detector</h1>
    <p>Real-time weed detection with YOLOv8</p>
    
    <div class="container">
        <div class="image-container">
            <h2>Original Image</h2>
            {% if latest_image %}
                <img src="{{ url_for('uploaded_file', filename=latest_image) }}" alt="Latest weed scan">
                <p>Image: {{ latest_image }}</p>
            {% else %}
                <div class="no-image">No images uploaded yet.</div>
            {% endif %}
        </div>
        
        <div class="image-container">
            <h2>Detection Result</h2>
            {% if detected_image %}
                <img src="{{ url_for('result_file', filename=detected_image) }}" alt="Detection result">
            {% elif latest_image %}
                <div class="no-image">Detection not performed yet.</div>
            {% else %}
                <div class="no-image">No images uploaded yet.</div>
            {% endif %}
        </div>
        
        <div class="data-container">
            <h2 class="detection-title">Weed Coordinates</h2>
            {% if detection_data %}
                <p>Timestamp: {{ detection_data.timestamp }}</p>
                {% if detection_data.detections %}
                    <table>
                        <tr>
                            <th>Confidence</th>
                            <th>X</th>
                            <th>Y</th>
                        </tr>
                        {% for detection in detection_data.detections %}
                        <tr>
                            <td>{{ detection.c }}</td>
                            <td>{{ detection.x }}</td>
                            <td>{{ detection.y }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                {% else %}
                    <p>No weeds detected in this image.</p>
                {% endif %}
            {% else %}
                <div class="no-image">No detection data available.</div>
            {% endif %}
            <p style="margin-top: 20px;">
                <a href="/coordinates" target="_blank">Download all coordinates as JSON</a>
            </p>
        </div>
    </div>
</body>
</html>
        ''')
    
    # Pre-load the model when starting the server
    load_model()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)