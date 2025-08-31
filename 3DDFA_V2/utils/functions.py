import cv2
import numpy as np
import torch

def cv_draw_landmark(img_ori, pts, box=None, color=(0, 255, 0), line_width=2):
    """Draw landmarks on image
    
    Args:
        img_ori (numpy.ndarray): Input image (BGR format)
        pts (numpy.ndarray): 2D landmarks with shape (n, 2)
        box (tuple, optional): Bounding box (x1, y1, x2, y2). Defaults to None.
        color (tuple, optional): Color in BGR format. Defaults to (0, 255, 0).
        line_width (int, optional): Line width. Defaults to 2.
        
    Returns:
        numpy.ndarray: Image with landmarks drawn
    """
    img = img_ori.copy()
    
    # Draw bounding box if provided
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw landmarks
    for (x, y) in pts.astype(np.int32):
        cv2.circle(img, (x, y), line_width, color, -1)
    
    return img

def get_suffix(filename):
    """Get the suffix of a filename"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]

def crop_img(img, roi_box):
    """Crop image according to the given roi box
    
    Args:
        img (numpy.ndarray): Input image
        roi_box (list or numpy.ndarray): [x1, y1, x2, y2]
        
    Returns:
        numpy.ndarray: Cropped image
    """
    h, w = img.shape[:2]
    
    # Convert roi_box to integers
    roi_box = [int(round(x)) for x in roi_box]
    
    # Clamp coordinates to image boundaries
    x1, y1, x2, y2 = roi_box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    
    # Crop the image
    cropped = img[y1:y2 + 1, x1:x2 + 1]
    
    return cropped

def to_tensor(img):
    """Convert numpy array to torch tensor
    
    Args:
        img (numpy.ndarray): Input image (H, W, C) in BGR format
        
    Returns:
        torch.Tensor: Output tensor (C, H, W) in RGB format
    """
    # Convert BGR to RGB and normalize to [0, 1]
    img = img[..., ::-1].astype('float32') / 255.
    
    # Convert HWC to CHW
    img = img.transpose((2, 0, 1))
    
    # Convert to tensor
    img = torch.from_numpy(img.copy())
    
    return img

def to_numpy(tensor):
    """Convert torch tensor to numpy array
    
    Args:
        tensor (torch.Tensor): Input tensor (C, H, W) in RGB format
        
    Returns:
        numpy.ndarray: Output array (H, W, C) in BGR format
    """
    # Convert to numpy and transpose CHW to HWC
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    
    # Convert RGB to BGR and scale to [0, 255]
    img = (img[..., ::-1] * 255).astype('uint8')
    
    return img
