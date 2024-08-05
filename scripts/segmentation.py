import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np
import cv2
import os

def segment_image(image_path, output_dir):
    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)
    
    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        mask = mask[0]
        mask_image = (mask * 255).astype(np.uint8)
        mask_image = cv2.resize(mask_image, (image.width, image.height))
        output_path = os.path.join(output_dir, f'mask_{i}.png')
        cv2.imwrite(output_path, mask_image)

        box = boxes[i].astype(int)
        cv2.rectangle(mask_image, (box[0], box[1]), (box[2], box[3]), 255, 2)

    return boxes, labels
