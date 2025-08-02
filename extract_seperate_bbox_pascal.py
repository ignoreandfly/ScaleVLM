import os
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

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

def parse_voc_annotation(xml_path):
    """Parse a single VOC XML annotation file"""
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
    
    objects = []
    
    # Parse all objects in the image
    for idx, obj in enumerate(root.findall('object')):
        class_name = obj.find('name').text
        
        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Calculate bounding box area
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_area = bbox_width * bbox_height
        
        # Calculate relative area percentage
        relative_area_percent = (bbox_area / img_area) * 100
        
        objects.append({
            'class_name': class_name,
            'bbox': (xmin, ymin, xmax, ymax),
            'bbox_area': bbox_area,
            'relative_area_percent': relative_area_percent,
            'size_category': categorize_size(relative_area_percent),
            'object_idx': idx
        })
    
    return image_id, filename, img_width, img_height, objects

def create_masked_image(image_path, target_bbox, all_bboxes):
    """Create image with target object visible and all OTHER objects blacked out"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Start with the original full image
    masked_image = image.copy()
    
    # Black out all OTHER bounding boxes (not the target one)
    for bbox in all_bboxes:
        if bbox != target_bbox:  # Don't black out the target object
            xmin, ymin, xmax, ymax = bbox
            masked_image[ymin:ymax, xmin:xmax] = 0  # Set to black
    
    return masked_image

def extract_and_process_objects(images_dir, annotations_dir, dataset_name, output_dir):
    """Extract individual objects from images and create masked versions"""
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, f"{dataset_name}_objects")
    os.makedirs(output_images_dir, exist_ok=True)
    
    results = []
    xml_files = list(Path(annotations_dir).glob('*.xml'))
    
    print(f"Processing {len(xml_files)} images from {dataset_name}...")
    
    for xml_file in xml_files:
        try:
            image_id, filename, img_width, img_height, objects = parse_voc_annotation(xml_file)
            
            # Construct image path
            image_path = os.path.join(images_dir, filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Extract all bounding boxes for this image
            all_bboxes = [obj['bbox'] for obj in objects]
            
            # Process each object
            for obj in objects:
                # Create unique filename for this object
                object_filename = f"{image_id}_{obj['object_idx']}_{obj['class_name']}_{obj['size_category']}.jpg"
                object_output_path = os.path.join(output_images_dir, object_filename)
                
                # Create masked image (only target object visible, others blacked out)
                masked_image = create_masked_image(image_path, obj['bbox'], all_bboxes)
                
                # Save the masked image
                cv2.imwrite(object_output_path, masked_image)
                
                # Record the result
                results.append({
                    'original_image_id': image_id,
                    'object_image_filename': object_filename,
                    'object_image_path': object_output_path,
                    'dataset': dataset_name,
                    'class_name': obj['class_name'],
                    'size_category': obj['size_category'],
                    'bbox_area': obj['bbox_area'],
                    'relative_area_percent': round(obj['relative_area_percent'], 2),
                    'bbox_coords': f"{obj['bbox'][0]},{obj['bbox'][1]},{obj['bbox'][2]},{obj['bbox'][3]}"
                })
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    return results

def main():
    """Main function to process both VOC datasets and generate individual object images"""
    
    # Define paths based on your directory structure
    train_val_images = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_train_val/VOC2012_train_val/JPEGImages"
    train_val_annotations = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_train_val/VOC2012_train_val/Annotations"
    test_images = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_test/VOC2012_test/JPEGImages"
    test_annotations = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_test/VOC2012_test/Annotations"
    
    # Output directory for extracted objects
    output_dir = "voc_extracted_objects"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Process train_val dataset
    print("Processing VOC2012_train_val dataset...")
    train_val_results = extract_and_process_objects(
        train_val_images, train_val_annotations, 'train_val', output_dir
    )
    all_results.extend(train_val_results)
    
    # Process test dataset
    print("Processing VOC2012_test dataset...")
    test_results = extract_and_process_objects(
        test_images, test_annotations, 'test', output_dir
    )
    all_results.extend(test_results)
    
    # Write results to CSV
    output_csv = os.path.join(output_dir, 'voc_individual_objects.csv')
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'original_image_id', 'object_image_filename', 'object_image_path',
            'dataset', 'class_name', 'size_category', 'bbox_area',
            'relative_area_percent', 'bbox_coords'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nProcessing complete!")
    print(f"Generated {len(all_results)} individual object images")
    print(f"Images saved to: {output_dir}/")
    print(f"Metadata saved to: {output_csv}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    size_counts = defaultdict(int)
    class_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    
    for result in all_results:
        size_counts[result['size_category']] += 1
        class_counts[result['class_name']] += 1
        dataset_counts[result['dataset']] += 1
    
    print(f"\nDataset distribution:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} objects")
    
    print(f"\nSize category distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size}: {count} objects")
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    print("Top 10 most frequent classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {class_name}: {count} objects")

# Additional utility function for CLIP benchmarking
def create_clip_benchmark_dataset(csv_path, size_category_filter=None):
    """
    Create a filtered dataset for CLIP benchmarking
    
    Args:
        csv_path: Path to the generated CSV file
        size_category_filter: Optional filter for specific size categories
        
    Returns:
        List of (image_path, class_name, size_category) tuples
    """
    benchmark_data = []
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if size_category_filter is None or row['size_category'] in size_category_filter:
                benchmark_data.append((
                    row['object_image_path'],
                    row['class_name'],
                    row['size_category']
                ))
    
    return benchmark_data

if __name__ == "__main__":
    main()
    
    # Example usage for CLIP benchmarking:
    # 
    # # Get all tiny objects for benchmarking
    # tiny_objects = create_clip_benchmark_dataset('voc_extracted_objects/voc_individual_objects.csv', ['tiny'])
    # 
    # # Get all objects of specific classes
    # all_objects = create_clip_benchmark_dataset('voc_extracted_objects/voc_individual_objects.csv')
    # person_objects = [obj for obj in all_objects if obj[1] == 'person']