import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# VOC class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Load CLIP model from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"  # or "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Configuration
DATA_ROOT = "/data/azfarm/siddhant/ICCV/dataset/VOC2012_clip"  # Update this path
# IMAGES_DIR = os.path.join(DATA_ROOT, "JPEGImages")
CSV_PATH = "/data/azfarm/siddhant/ICCV/voc_object_sizes.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)
# test_data = df[df['dataset'] == 'test'].copy()

print(f"Total samples: {len(df)}")
print(f"Size distribution: {df['size_category'].value_counts().to_dict()}")
# Group by size category
size_groups = df.groupby('size_category')

results = {}
all_predictions = []

for size_category, group in size_groups:
    print(f"\nEvaluating {size_category} objects ({len(group)} samples)...")
    
    correct_top1 = 0
    correct_top5 = 0
    total_confidence = 0
    size_predictions = []
    
    # Process in batches for efficiency
    batch_size = 32
    for i in tqdm(range(0, len(group), batch_size), desc=f"Processing {size_category}"):
        batch = group.iloc[i:i+batch_size]
        
        # Load batch images
        batch_images = []
        batch_labels = []
        valid_indices = []
        
        for idx, (_, row) in enumerate(batch.iterrows()):
            image_path = os.path.join(DATA_ROOT, f"{row['image_id']}.jpg")
            
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    batch_images.append(image)
                    batch_labels.append(row['class_name'])
                    valid_indices.append(idx)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
        
        if not batch_images:
            continue
            
        # Create text prompts
        text_prompts = [f"a photo of a {cls}" for cls in VOC_CLASSES]
        
        # Process batch
        inputs = processor(
            text=text_prompts, 
            images=batch_images, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=1).cpu().numpy()
        
        # Calculate metrics for each image in batch
        for img_idx, (image_probs, true_class) in enumerate(zip(probs, batch_labels)):
            try:
                true_class_idx = VOC_CLASSES.index(true_class)
            except ValueError:
                print(f"Unknown class: {true_class}")
                continue
                
            # Get top predictions
            top5_indices = image_probs.argsort()[-5:][::-1]
            
            # Update counters
            if top5_indices[0] == true_class_idx:
                correct_top1 += 1
            if true_class_idx in top5_indices:
                correct_top5 += 1
                
            confidence = image_probs[true_class_idx]
            total_confidence += confidence
            
            # Store prediction details
            prediction_data = {
                'image_id': batch.iloc[valid_indices[img_idx]]['image_id'],
                'true_class': true_class,
                'size_category': size_category,
                'predicted_class': VOC_CLASSES[top5_indices[0]],
                'confidence': confidence,
                'top1_correct': top5_indices[0] == true_class_idx,
                'top5_correct': true_class_idx in top5_indices,
                'top5_classes': [VOC_CLASSES[idx] for idx in top5_indices],
                'top5_probs': image_probs[top5_indices].tolist()
            }
            size_predictions.append(prediction_data)
    
    # Calculate final metrics for this size category
    total_samples = len(size_predictions)
    if total_samples > 0:
        results[size_category] = {
            'samples': total_samples,
            'top1_accuracy': correct_top1 / total_samples,
            'top5_accuracy': correct_top5 / total_samples,
            'mean_confidence': total_confidence / total_samples,
            'predictions': size_predictions
        }
        
        print(f"{size_category.capitalize()} Results:")
        print(f"  Samples: {total_samples}")
        print(f"  Top-1 Accuracy: {results[size_category]['top1_accuracy']:.3f}")
        print(f"  Top-5 Accuracy: {results[size_category]['top5_accuracy']:.3f}")
        print(f"  Mean Confidence: {results[size_category]['mean_confidence']:.3f}")
    
    all_predictions.extend(size_predictions)

# Print overall results
print("\n" + "="*50)
print("FINAL RESULTS BY SIZE CATEGORY")
print("="*50)

size_order = ['tiny', 'small', 'medium', 'large', 'huge']
for size in size_order:
    if size in results:
        metrics = results[size]
        print(f"{size.upper():>8}: "
              f"Top-1: {metrics['top1_accuracy']:.3f} | "
              f"Top-5: {metrics['top5_accuracy']:.3f} | "
              f"Conf: {metrics['mean_confidence']:.3f} | "
              f"N: {metrics['samples']}")

# Save detailed results
results_df = pd.DataFrame(all_predictions)
results_df.to_csv('clip_size_benchmark_results.csv', index=False)
print(f"\nDetailed results saved to: clip_size_benchmark_results.csv")

# Save summary results
summary_data = []
for size, metrics in results.items():
    summary_data.append({
        'size_category': size,
        'samples': metrics['samples'],
        'top1_accuracy': metrics['top1_accuracy'],
        'top5_accuracy': metrics['top5_accuracy'],
        'mean_confidence': metrics['mean_confidence']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('clip_size_benchmark_summary.csv', index=False)
print(f"Summary results saved to: clip_size_benchmark_summary.csv")