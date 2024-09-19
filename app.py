import sys
import os

# Ensure Detectron2 is in the Python path
sys.path.insert(0, os.path.abspath('./detectron2'))

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Flask app!"

# Ensure the uploads directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Set up Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file('detectron2_model/config.yaml')
cfg.MODEL.WEIGHTS = os.path.join('detectron2_model', 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Load metadata
with open('detectron2_model/train_metadata(1).json', "r") as f:
    loaded_metadata = json.load(f)
MetadataCatalog.get("my_dataset_val").set(thing_classes=loaded_metadata["thing_classes"])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file
    filepath = os.path.join('static/uploads/', file.filename)
    file.save(filepath)
    
    # Load the image
    im = cv2.imread(filepath)
    if im is None:
        return jsonify({"error": "Error reading image file"}), 400

    # Get predictions
    outputs = predictor(im)
    instances = outputs["instances"]
    
    pred_classes = instances.pred_classes
    class_names = MetadataCatalog.get("my_dataset_val").get("thing_classes", [])
    
    # Prepare output
    results = {
        "predicted_classes": [class_names[int(cls)] for cls in pred_classes.tolist()],
        "num_classes": len(class_names)
    }

    # Calculate and print the mask areas
    try:
        leaf_index = (pred_classes == 0).nonzero(as_tuple=True)[0]
        blight_index = (pred_classes == 1).nonzero(as_tuple=True)[0]

        leaf_mask = instances.pred_masks[leaf_index]
        blight_mask = instances.pred_masks[blight_index]

        leaf_mask_np = leaf_mask[0].cpu().numpy()
        leaf_area = leaf_mask_np.sum()

        blight_mask_np = blight_mask[0].cpu().numpy() if len(blight_mask) > 0 else np.zeros_like(leaf_mask_np)
        blight_area = blight_mask_np.sum()

        infec = blight_area / leaf_area if leaf_area > 0 else 0

        # Determine the infection status
        if infec > 0.02 and infec < 0.15:
            status = "Early Blight"
        elif infec >= 0.15:
            status = "Late Blight"
        elif infec < 0.01:
            status = "Healthy"
        else:
            status = "Unknown"

        # Add these details to the results (convert values to standard Python types)
        results["leaf_area"] = int(leaf_area)
        results["blight_area"] = int(blight_area)
        results["infection_rate"] = float(infec)
        results["infection_status"] = status

    except Exception as e:
        results["error"] = f"Error calculating areas: {str(e)}"
    
    # Visualize the predictions
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("my_dataset_val"),
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances.to("cpu"))
    
    # Save visualization
    vis_path = os.path.join('static/uploads/', 'visualization.jpg')
    cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

    # Construct URL for visualization
    vis_url = f"/static/uploads/visualization.jpg"

    results["visualization_url"] = vis_url

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

