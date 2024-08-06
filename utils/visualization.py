import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_segmented_image(image, boxes, labels):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            x_min, y_min - 10,
            f'{label}',
            bbox=dict(facecolor='white', alpha=0.5),
            fontsize=12,
            color='black'
        )

    plt.axis('off')
    plt.savefig('output/segmented_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
