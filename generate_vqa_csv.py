import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict
import os

class VQAConverter:
    def __init__(self, csv_path):
        """
        Initialize VQA converter with CSV data
        
        Args:
            csv_path: Path to the CSV file with extracted objects
        """
        self.df = pd.read_csv(csv_path)
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # Base paths for extracted object images
        self.base_paths = {
            'train_val': '/data/azfarm/voc_extracted_objects/train_val_objects',
            'test': '/data/azfarm/voc_extracted_objects/test_objects'
        }
        
    def generate_positive_vqa_questions(self):
        """
        Generate one positive VQA question per image (number of questions = number of images)
        
        Returns:
            List of VQA question dictionaries - one per image
        """
        vqa_questions = []
        
        # Generate one question for each extracted object image
        for _, row in self.df.iterrows():
            object_image_filename = row['object_image_filename']
            class_name = row['class_name']
            size_category = row['size_category']
            dataset = row['dataset']
            original_image_id = row['original_image_id']
            
            # Construct full path to extracted object image
            image_path = os.path.join(self.base_paths[dataset], object_image_filename)
            
            # Positive question - the object IS present (it's the extracted object)
            question = f"Is there a {class_name} in this image?"
            
            vqa_questions.append({
                'image_id': object_image_filename.replace('.jpg', ''),  # Use object filename as ID
                'image_path': image_path,  # Full path to extracted object image
                'original_image_id': original_image_id,
                'dataset': dataset,
                'question': question,
                'answer': 'Yes',
                'target_class': class_name,
                'target_size': size_category,
                'question_type': 'presence_positive_extracted'
            })
        
        # Shuffle questions
        random.shuffle(vqa_questions)
        return vqa_questions
    
    def save_vqa_dataset(self, vqa_questions, output_path, format='json'):
        """Save VQA dataset to file"""
        
        if format == 'json':
            # Save as JSON (standard VQA format)
            with open(output_path, 'w') as f:
                json.dump(vqa_questions, f, indent=2)
                
        elif format == 'csv':
            # Save as CSV
            df = pd.DataFrame(vqa_questions)
            df.to_csv(output_path, index=False)
            
        print(f"Saved {len(vqa_questions)} VQA questions to {output_path}")
        
    def generate_analysis_report(self, vqa_questions):
        """Generate analysis report of the VQA dataset"""
        
        print("\n" + "="*80)
        print("VQA DATASET ANALYSIS REPORT (POSITIVE QUESTIONS ONLY)")
        print("="*80)
        
        df = pd.DataFrame(vqa_questions)
        
        print(f"\nTOTAL QUESTIONS: {len(vqa_questions)} (1 per image)")
        print(f"TOTAL IMAGES: {len(vqa_questions)}")
        
        # Answer distribution (should be 100% Yes)
        print(f"\nANSWER DISTRIBUTION:")
        answer_dist = df['answer'].value_counts()
        for answer, count in answer_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")
        
        # Dataset distribution
        print(f"\nDATASET DISTRIBUTION:")
        dataset_dist = df['dataset'].value_counts()
        for dataset, count in dataset_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {dataset}: {count} ({percentage:.1f}%)")
        
        # Class distribution
        print(f"\nCLASS DISTRIBUTION:")
        class_dist = df['target_class'].value_counts()
        for class_name, count in class_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Size distribution
        print(f"\nSIZE CATEGORY DISTRIBUTION:")
        size_dist = df['target_size'].value_counts()
        size_order = ['tiny', 'small', 'medium', 'large', 'huge']
        for size in size_order:
            if size in size_dist:
                count = size_dist[size]
                percentage = (count / len(df)) * 100
                print(f"  {size}: {count} ({percentage:.1f}%)")

def main():
    """Main function to convert CSV to simple VQA dataset - one question per image"""
    
    # Configuration
    CSV_PATH = "/data/azfarm/voc_extracted_objects/voc_individual_objects.csv"
    OUTPUT_DIR = "/data/azfarm/siddhant/ICCV/simple_vqa_dataset"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize converter
    print("Loading CSV data...")
    converter = VQAConverter(CSV_PATH)
    
    print(f"Found {len(converter.df)} extracted object images")
    print("Generating 1 positive VQA question per image")
    
    # Generate positive VQA questions (1 per image)
    print("\nGenerating positive VQA questions...")
    
    vqa_questions = converter.generate_positive_vqa_questions()
    
    # Save datasets
    json_path = os.path.join(OUTPUT_DIR, "simple_vqa_questions.json")
    csv_path = os.path.join(OUTPUT_DIR, "simple_vqa_questions.csv")
    
    converter.save_vqa_dataset(vqa_questions, json_path, format='json')
    converter.save_vqa_dataset(vqa_questions, csv_path, format='csv')
    
    # Generate analysis
    converter.generate_analysis_report(vqa_questions)
    
    print(f"\nVQA dataset generation complete!")
    print(f"Generated {len(vqa_questions)} questions for {len(vqa_questions)} images")
    print(f"Files saved to: {OUTPUT_DIR}/")
    
    # Save a sample for inspection
    sample_path = os.path.join(OUTPUT_DIR, "sample_questions.json")
    sample_questions = random.sample(vqa_questions, min(10, len(vqa_questions)))
    with open(sample_path, 'w') as f:
        json.dump(sample_questions, f, indent=2)
    print(f"Sample questions saved to: {sample_path}")
    
    # Verify some image paths exist
    print(f"\nVerifying image paths...")
    sample_paths = [q['image_path'] for q in sample_questions[:5]]
    for path in sample_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")
    
    return vqa_questions

if __name__ == "__main__":
    main()