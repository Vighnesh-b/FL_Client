import numpy as np
import re, os
from utils.predict_eval_utils import predict, evaluate_per_slice
from models.unetr_model import get_unetr
import torch
from monai.transforms import DivisiblePad

def predict_and_evaluate_mask(image_path, mask_path=None, model_path=None, device=None):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model checkpoint
    if not model_path:
        checkpoint_dir = "client_checkpoints/"
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        files = os.listdir(checkpoint_dir)
        round_files = [f for f in files if re.match(r"round_(\d+)\.pth$", f)]
        
        if not round_files:
            raise FileNotFoundError("No checkpoint files found in client_checkpoints/")
            
        latest = max(round_files, key=lambda f: int(re.search(r"(\d+)", f).group()))
        model_path = os.path.join(checkpoint_dir, latest)
    
    # Check if files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if mask_path and not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load model
    model = get_unetr(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    
    # Load and preprocess image
    img_array = np.load(image_path)
    image = np.transpose(img_array, (3, 2, 0, 1))  # Adjust based on your data format
    image = torch.from_numpy(image).float().to(device)  # Move to device
    
    pad = DivisiblePad(k=16)
    image = pad(image)
    
    # Predict
    with torch.no_grad():  # Disable gradient computation
        pred_mask = predict(model, image, device)
    
    # If no ground truth, return prediction only
    if not mask_path:
        return pred_mask, None
    
    # Load and preprocess ground truth mask
    mask_array = np.load(mask_path)
    mask = np.transpose(mask_array, (2, 0, 1))  # Check if this matches your data
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).float().to(device)  # Move to device
    mask = pad(mask)
    
    # Evaluate
    dice_scores = evaluate_per_slice(pred_mask, mask)
    
    return pred_mask, dice_scores, image, mask
