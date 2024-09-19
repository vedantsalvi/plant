import gdown
import os

# Google Drive file ID
file_id = "1-SFDWKaSUwcsnPg8STLRZxI_gqNEWDzU"
output_dir = "detectron2_model"
output_file = os.path.join(output_dir, "model_final.pth")

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download the model if it doesn't exist
if not os.path.exists(output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    print(f"Downloaded model to {output_file}")
else:
    print("Model already exists.")
