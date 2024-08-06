import numpy as np

def postprocess_segmentation(segmented_data):
    # Convert masks to binary masks
    masks = segmented_data['masks']
    masks = np.where(masks > 0.5, 1, 0)
    return {
        'masks': masks,
        'boxes': segmented_data['boxes']
    }
