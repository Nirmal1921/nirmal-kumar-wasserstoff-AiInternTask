import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np

class SegmentationModel:
    def __init__(self):
        # Load the pre-trained Mask R-CNN model
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment(self, image: Image.Image):
        # Convert image to tensor and apply transformations
        image_tensor = self.transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Process masks
        masks = outputs[0]['masks'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        # Create an empty array to store segmented regions
        segmented_regions = []
        for i in range(len(masks)):
            if scores[i] > 0.5:  # Threshold to filter out low-confidence predictions
                mask = masks[i][0] > 0.5  # Apply threshold to create binary mask
                segmented_regions.append({
                    'label': int(labels[i]),
                    'mask': mask
                })

        return segmented_regions
