import os
import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
from pathlib import Path

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
    
    # Get image filename (without extension for image_id)
    filename = root.find('filename').text
    image_id = os.path.splitext(filename)[0]
    
    objects = []
    
    # Parse all objects in the image
    for obj in root.findall('object'):
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
            'bbox_area': bbox_area,
            'relative_area_percent': relative_area_percent,
            'size_category': categorize_size(relative_area_percent)
        })
    
    return image_id, img_area, objects

def process_voc_dataset(annotations_dir, dataset_name):
    """Process all annotations in a VOC dataset directory"""
    results = []
    
    # Find all XML files in the annotations directory
    xml_files = list(Path(annotations_dir).glob('*.xml'))
    
    print(f"Processing {len(xml_files)} annotations from {dataset_name}...")
    
    for xml_file in xml_files:
        try:
            image_id, img_area, objects = parse_voc_annotation(xml_file)
            
            # Group objects by class and find the largest instance of each class
            class_objects = defaultdict(list)
            for obj in objects:
                class_objects[obj['class_name']].append(obj)
            
            # For each class, take the largest instance
            for class_name, class_instances in class_objects.items():
                # Find instance with largest area
                largest_instance = max(class_instances, key=lambda x: x['bbox_area'])
                
                results.append({
                    'image_id': image_id,
                    'dataset': dataset_name,
                    'class_name': class_name,
                    'size_category': largest_instance['size_category']
                })
                
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    return results

def main():
    """Main function to process both VOC datasets and generate CSV"""
    
    # Define paths based on your directory structure
    train_val_annotations = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_train_val/VOC2012_train_val/Annotations"
    test_annotations = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_test/VOC2012_test/Annotations"
    
    all_results = []
    
    # Process train_val dataset
    print("Processing VOC2012_train_val dataset...")
    train_val_results = process_voc_dataset(train_val_annotations, 'train_val')
    all_results.extend(train_val_results)
    
    # Process test dataset
    print("Processing VOC2012_test dataset...")
    test_results = process_voc_dataset(test_annotations, 'test')
    all_results.extend(test_results)
    
    # Write results to CSV
    output_file = '/data/azfarm/siddhant/ICCV/voc_object_sizes.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_id', 'dataset', 'class_name', 'size_category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nProcessing complete!")
    print(f"Generated {output_file} with {len(all_results)} entries")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    size_counts = defaultdict(int)
    class_counts = defaultdict(int)
    
    for result in all_results:
        size_counts[result['size_category']] += 1
        class_counts[result['class_name']] += 1
    
    print("\nSize category distribution:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size}: {count}")
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    print("Top 10 most frequent classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {class_name}: {count}")

if __name__ == "__main__":
    main()