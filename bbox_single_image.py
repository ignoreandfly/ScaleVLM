import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random

def categorize_size(relative_area_percent):
    """Categorize object size based on relative area percentage"""
    if relative_area_percent < 1.0:
        return 'tiny'
    elif relative_area_percent < 5.0:
        return 'small'  
    elif relative_area_percent < 15.0:
        return 'medium'
    elif relative_area_percent < 40.0:
        return 'large'
    else:
        return 'huge'

def get_size_color(size_category):
    """Get color for each size category"""
    colors = {
        'tiny': 'red',
        'small': 'orange', 
        'medium': 'yellow',
        'large': 'green',
        'huge': 'blue'
    }
    return colors.get(size_category, 'white')

def parse_and_visualize_image(image_path, xml_path):
    """Parse XML annotation and visualize bounding boxes on image"""
    
    # Parse XML annotation
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    img_area = img_width * img_height
    
    # Get image filename
    filename = root.find('filename').text
    image_id = os.path.splitext(filename)[0]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Parse all objects and draw bounding boxes
    objects_info = []
    
    for idx, obj in enumerate(root.findall('object')):
        class_name = obj.find('name').text
        
        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Calculate bounding box area and relative size
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_area = bbox_width * bbox_height
        relative_area_percent = (bbox_area / img_area) * 100
        size_category = categorize_size(relative_area_percent)
        
        # Store object info
        objects_info.append({
            'idx': idx,
            'class_name': class_name,
            'bbox': (xmin, ymin, xmax, ymax),
            'size_category': size_category,
            'relative_area': relative_area_percent
        })
        
        # Draw bounding box
        color = get_size_color(size_category)
        rect = Rectangle((xmin, ymin), bbox_width, bbox_height, 
                        linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label with class name and size
        label = f"{idx}: {class_name} ({size_category})"
        ax.text(xmin, ymin-10, label, fontsize=10, color=color, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Set title and remove axes
    ax.set_title(f"Image: {image_id} | Objects: {len(objects_info)}", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = []
    for size in ['tiny', 'small', 'medium', 'large', 'huge']:
        legend_elements.append(patches.Patch(color=get_size_color(size), label=f'{size.capitalize()}'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    output_dir = "output_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{image_id}_viz.jpg")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    # Print detailed information
    print(f"\nImage: {image_id}")
    print(f"Dimensions: {img_width} x {img_height} (Area: {img_area:,} pixels)")
    print(f"Total objects: {len(objects_info)}")
    print("\nObject Details:")
    print("-" * 80)
    
    for obj in objects_info:
        xmin, ymin, xmax, ymax = obj['bbox']
        print(f"[{obj['idx']}] {obj['class_name']:<12} | "
              f"Size: {obj['size_category']:<6} | "
              f"BBox: ({xmin:3d},{ymin:3d},{xmax:3d},{ymax:3d}) | "
              f"Area: {obj['relative_area']:.1f}%")
    
    return objects_info

def visualize_specific_image(image_id, dataset='train_val'):
    """Visualize a specific image by ID"""
    
    # Construct paths
    if dataset == 'train_val':
       image_path = f"/data/azfarm/siddhant/ICCV/dataset/VOC2012_train_val/VOC2012_train_val/JPEGImages/{image_id}.jpg"
       xml_path = f"/data/azfarm/siddhant/ICCV/dataset/VOC2012_train_val/VOC2012_train_val/Annotations/{image_id}.xml"

    else:  # test
        image_path = f"/data/azfarm/siddhant/ICCV/dataset/VOC2012_test//JPEGImages/{image_id}.jpg"
        xml_path = f"/data/azfarm/siddhant/ICCV/dataset/VOC2012_test/Annotations/{image_id}.xml"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    if not os.path.exists(xml_path):
        print(f"Annotation not found: {xml_path}")
        return
    
    # Visualize
    return parse_and_visualize_image(image_path, xml_path)

def batch_visualize_images(image_ids, dataset='train_val'):
    """Visualize multiple images"""
    for image_id in image_ids:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {image_id}")
        print('='*80)
        visualize_specific_image(image_id, dataset)
        
        # Ask user if they want to continue (optional)
        if len(image_ids) > 1:
            response = input(f"\nPress Enter to continue to next image, or 'q' to quit: ")
            if response.lower() == 'q':
                break

def explore_dataset_samples(dataset='train_val', num_samples=5):
    """Randomly sample and visualize images from dataset"""
    
    # Get annotation directory
    if dataset == 'train_val':
        annotations_dir = "VOC2012_train_val/Annotations"
    else:
        annotations_dir = "VOC2012_test/Annotations"
    
    # Get all XML files
    xml_files = [f.stem for f in Path(annotations_dir).glob('*.xml')]
    
    if not xml_files:
        print(f"No annotation files found in {annotations_dir}")
        return
    
    # Sample random images
    sample_images = random.sample(xml_files, min(num_samples, len(xml_files)))
    
    print(f"Randomly selected {len(sample_images)} images from {dataset} dataset:")
    for img_id in sample_images:
        print(f"  - {img_id}")
    
    batch_visualize_images(sample_images, dataset)

# Example usage
if __name__ == "__main__":
    # Visualize the specific image you showed
    print("Visualizing image: 2007_000170")
    visualize_specific_image("2007_000170", "train_val")  # or "test" if it's in test set
    
    # You can also visualize multiple specific images
    # batch_visualize_images(["2007_000256", "2007_000027"], "train_val")
    
    # Or explore random samples from your dataset
    # explore_dataset_samples("train_val", num_samples=3)