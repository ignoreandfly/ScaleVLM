import os
import csv
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

class HFCLIPObjectSizeBenchmark:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize CLIP benchmark using Hugging Face Transformers
        
        Args:
            model_name: HF CLIP model to use 
                       ("openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", 
                        "openai/clip-vit-large-patch14", etc.)
            device: Device to run on (auto-detect if None)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        
        # Set model to eval mode
        self.model.eval()
        
        # VOC class names (20 classes)
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # Create text prompts
        self.text_prompts = [f"a photo of a {class_name}" for class_name in self.voc_classes]
        
        # Precompute text embeddings for all classes
        self.text_embeddings = self._precompute_text_embeddings()
        
    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all VOC classes"""
        print("Precomputing text embeddings for VOC classes...")
        
        # Process text inputs
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**text_inputs)
            # Normalize embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings
    
    def predict_single_image(self, image_path):
        """
        Predict class for a single image using Hugging Face CLIP
        
        Returns:
            tuple: (predicted_class, confidence_scores, top5_predictions, all_probabilities)
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Get image embedding
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**image_inputs)
                # Normalize embeddings
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                
                # Calculate similarities with text embeddings
                similarities = torch.matmul(image_embeddings, self.text_embeddings.T).squeeze(0)
                
                # Convert to probabilities
                probabilities = torch.softmax(similarities, dim=-1)
            
            # Get predictions
            top5_probs, top5_indices = probabilities.topk(5)
            
            predicted_idx = top5_indices[0].item()
            predicted_class = self.voc_classes[predicted_idx]
            confidence = top5_probs[0].item()
            
            # Top 5 predictions
            top5_predictions = [
                (self.voc_classes[idx.item()], prob.item()) 
                for idx, prob in zip(top5_indices, top5_probs)
            ]
            
            return predicted_class, confidence, top5_predictions, probabilities.cpu().numpy()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0, [], np.zeros(len(self.voc_classes))
    
    def benchmark_dataset(self, csv_path, output_dir="hf_clip_benchmark_results"):
        """
        Benchmark CLIP on the extracted object dataset
        
        Args:
            csv_path: Path to CSV file with extracted objects
            output_dir: Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset loaded: {len(df)} objects")
        print(f"Size distribution: {df['size_category'].value_counts().to_dict()}")
        print(f"Class distribution: {df['class_name'].value_counts().head(10).to_dict()}")
        
        # Initialize results storage
        results = []
        
        # Process each image
        print("\nRunning CLIP predictions...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_path = row['object_image_path']
            true_class = row['class_name']
            size_category = row['size_category']
            dataset_split = row['dataset']
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Get CLIP prediction
            pred_class, confidence, top5, all_probs = self.predict_single_image(image_path)
            
            if pred_class is None:
                continue
            
            # Calculate metrics
            is_correct = (pred_class == true_class)
            top5_correct = true_class in [pred[0] for pred in top5]
            
            # Store result
            result = {
                'original_image_id': row['original_image_id'],
                'object_image_filename': row['object_image_filename'],
                'dataset': dataset_split,
                'true_class': true_class,
                'predicted_class': pred_class,
                'size_category': size_category,
                'confidence': confidence,
                'is_correct': is_correct,
                'top5_correct': top5_correct,
                'bbox_area': row['bbox_area'],
                'relative_area_percent': row['relative_area_percent']
            }
            
            # Add top-5 predictions
            for i, (cls, prob) in enumerate(top5):
                result[f'top{i+1}_class'] = cls
                result[f'top{i+1}_prob'] = prob
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        detailed_results_path = os.path.join(output_dir, f"detailed_results_{model_safe_name}_{timestamp}.csv")
        results_df.to_csv(detailed_results_path, index=False)
        
        print(f"\nDetailed results saved to: {detailed_results_path}")
        
        # Generate analysis
        self._analyze_results(results_df, output_dir, timestamp)
        
        return results_df
    
    def _analyze_results(self, results_df, output_dir, timestamp):
        """Generate comprehensive analysis of results"""
        
        print("\n" + "="*80)
        print("HUGGING FACE CLIP BENCHMARK ANALYSIS")
        print("="*80)
        
        # Overall metrics
        overall_accuracy = results_df['is_correct'].mean()
        overall_top5_accuracy = results_df['top5_correct'].mean()
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Model: {self.model_name}")
        print(f"Total objects tested: {len(results_df)}")
        print(f"Top-1 Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        print(f"Top-5 Accuracy: {overall_top5_accuracy:.3f} ({overall_top5_accuracy*100:.1f}%)")
        
        # Performance by size category
        print(f"\nPERFORMANCE BY SIZE CATEGORY:")
        size_analysis = results_df.groupby('size_category').agg({
            'is_correct': ['count', 'mean'],
            'top5_correct': 'mean',
            'confidence': 'mean'
        }).round(3)
        
        size_analysis.columns = ['Count', 'Top1_Accuracy', 'Top5_Accuracy', 'Avg_Confidence']
        
        # Sort by size category order
        size_order = ['tiny', 'small', 'medium', 'large', 'huge']
        size_analysis = size_analysis.reindex([s for s in size_order if s in size_analysis.index])
        print(size_analysis)
        
        # Highlight the key finding
        print(f"\n{'='*60}")
        print("KEY FINDINGS - VLM PERFORMANCE BY OBJECT SIZE:")
        print('='*60)
        for size in size_order:
            if size in size_analysis.index:
                acc = size_analysis.loc[size, 'Top1_Accuracy']
                count = size_analysis.loc[size, 'Count']
                print(f"{size.upper():>6} objects: {acc:.1%} accuracy ({count:>4} samples)")
        
        # Performance by class
        print(f"\nTOP 10 CLASSES BY ACCURACY:")
        class_analysis = results_df.groupby('true_class').agg({
            'is_correct': ['count', 'mean'],
            'confidence': 'mean'
        }).round(3)
        class_analysis.columns = ['Count', 'Accuracy', 'Avg_Confidence']
        class_analysis = class_analysis.sort_values('Accuracy', ascending=False)
        print(class_analysis.head(10))
        
        print(f"\nWORST 10 CLASSES BY ACCURACY:")
        print(class_analysis.tail(10))
        
        # Performance by dataset split
        print(f"\nPERFORMANCE BY DATASET SPLIT:")
        dataset_analysis = results_df.groupby('dataset').agg({
            'is_correct': ['count', 'mean'],
            'top5_correct': 'mean'
        }).round(3)
        dataset_analysis.columns = ['Count', 'Top1_Accuracy', 'Top5_Accuracy']
        print(dataset_analysis)
        
        # Size vs Class interaction analysis
        print(f"\nSIZE × CLASS INTERACTION (Top classes with multiple size categories):")
        size_class_analysis = results_df.groupby(['true_class', 'size_category']).agg({
            'is_correct': ['count', 'mean']
        }).round(3)
        size_class_analysis.columns = ['Count', 'Accuracy']
        
        # Show classes that appear in multiple sizes
        classes_multi_size = results_df.groupby('true_class')['size_category'].nunique()
        multi_size_classes = classes_multi_size[classes_multi_size > 1].index
        
        for cls in multi_size_classes[:5]:  # Show top 5
            print(f"\n{cls.upper()}:")
            cls_data = size_class_analysis.loc[cls]
            if isinstance(cls_data, pd.DataFrame):
                print(cls_data)
        
        # Save summary statistics
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_objects': len(results_df),
            'overall_top1_accuracy': float(overall_accuracy),
            'overall_top5_accuracy': float(overall_top5_accuracy),
            'size_category_performance': size_analysis.to_dict(),
            'class_performance': class_analysis.to_dict(),
            'dataset_performance': dataset_analysis.to_dict()
        }
        
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        summary_path = os.path.join(output_dir, f"summary_{model_safe_name}_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_visualizations(results_df, output_dir, timestamp)
        
        print(f"\nSummary saved to: {summary_path}")
    
    def _create_visualizations(self, results_df, output_dir, timestamp):
        """Create visualization plots"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a larger figure with more subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'CLIP Object Size Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Size Category
        ax1 = axes[0, 0]
        size_order = ['tiny', 'small', 'medium', 'large', 'huge']
        size_acc = results_df.groupby('size_category')['is_correct'].mean()
        size_counts = results_df.groupby('size_category').size()
        
        # Reorder according to size_order
        size_acc_ordered = [size_acc.get(size, 0) for size in size_order if size in size_acc.index]
        size_counts_ordered = [size_counts.get(size, 0) for size in size_order if size in size_counts.index]
        size_labels = [size for size in size_order if size in size_acc.index]
        
        bars = ax1.bar(range(len(size_acc_ordered)), size_acc_ordered, 
                      color=['red', 'orange', 'gold', 'green', 'blue'][:len(size_acc_ordered)])
        ax1.set_xlabel('Size Category')
        ax1.set_ylabel('Top-1 Accuracy')
        ax1.set_title('Accuracy by Object Size')
        ax1.set_xticks(range(len(size_labels)))
        ax1.set_xticklabels(size_labels, rotation=45)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, size_counts_ordered)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        ax1.set_ylim(0, 1)
        
        # 2. Confidence by Size Category
        ax2 = axes[0, 1]
        sns.boxplot(data=results_df, x='size_category', y='confidence', ax=ax2, order=size_order)
        ax2.set_title('Confidence Distribution by Size')
        ax2.set_xlabel('Size Category')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Top-5 vs Top-1 Accuracy by Size
        ax3 = axes[0, 2]
        size_metrics = results_df.groupby('size_category').agg({
            'is_correct': 'mean',
            'top5_correct': 'mean'
        })
        size_metrics_ordered = size_metrics.reindex([s for s in size_order if s in size_metrics.index])
        
        x = np.arange(len(size_metrics_ordered))
        width = 0.35
        ax3.bar(x - width/2, size_metrics_ordered['is_correct'], width, label='Top-1', alpha=0.8)
        ax3.bar(x + width/2, size_metrics_ordered['top5_correct'], width, label='Top-5', alpha=0.8)
        ax3.set_xlabel('Size Category')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Top-1 vs Top-5 Accuracy by Size')
        ax3.set_xticks(x)
        ax3.set_xticklabels(size_metrics_ordered.index, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Accuracy by Class (top 15)
        ax4 = axes[1, 0]
        class_acc = results_df.groupby('true_class')['is_correct'].mean().sort_values(ascending=True)
        top_classes = class_acc.tail(15)
        
        ax4.barh(range(len(top_classes)), top_classes.values)
        ax4.set_xlabel('Top-1 Accuracy')
        ax4.set_ylabel('Object Class')
        ax4.set_title('Top 15 Classes by Accuracy')
        ax4.set_yticks(range(len(top_classes)))
        ax4.set_yticklabels(top_classes.index, fontsize=8)
        
        # 5. Size vs Accuracy Scatter with Class Info
        ax5 = axes[1, 1]
        colors = {'tiny': 'red', 'small': 'orange', 'medium': 'gold', 'large': 'green', 'huge': 'blue'}
        
        for size in results_df['size_category'].unique():
            size_data = results_df[results_df['size_category'] == size]
            ax5.scatter(size_data['relative_area_percent'], size_data['is_correct'], 
                       c=colors.get(size, 'gray'), label=size, alpha=0.6, s=30)
        
        ax5.set_xlabel('Relative Area (%)')
        ax5.set_ylabel('Accuracy (1=Correct, 0=Wrong)')
        ax5.set_title('Individual Predictions: Size vs Accuracy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Class Distribution by Size
        ax6 = axes[1, 2]
        size_class_counts = pd.crosstab(results_df['size_category'], results_df['true_class'])
        
        # Show top classes only
        top_classes_overall = results_df['true_class'].value_counts().head(8).index
        size_class_subset = size_class_counts[top_classes_overall]
        
        size_class_subset.plot(kind='bar', stacked=True, ax=ax6, colormap='tab20')
        ax6.set_title('Object Distribution by Size & Class')
        ax6.set_xlabel('Size Category')
        ax6.set_ylabel('Number of Objects')
        ax6.tick_params(axis='x', rotation=45)
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        plot_path = os.path.join(output_dir, f"analysis_plots_{model_safe_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: {plot_path}")

def main():
    """Main benchmarking function"""
    
    # Configuration - Popular HF CLIP models
    MODEL_NAME = "openai/clip-vit-base-patch32"  # Change to test different models
    # Other options: "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"
    
    CSV_PATH = "voc_extracted_objects/voc_individual_objects.csv"
    OUTPUT_DIR = "hf_clip_benchmark_results"
    
    # Initialize benchmark
    benchmark = HFCLIPObjectSizeBenchmark(model_name=MODEL_NAME)
    
    # Run benchmark
    results_df = benchmark.benchmark_dataset(CSV_PATH, OUTPUT_DIR)
    
    print(f"\nBenchmarking complete! Results saved to: {OUTPUT_DIR}")
    
    return results_df

# Utility functions for specific analyses
def compare_hf_models(csv_path, models=None):
    """Compare multiple HuggingFace CLIP models"""
    if models is None:
        models = [
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-base-patch16", 
            "openai/clip-vit-large-patch14"
        ]
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING MODEL: {model_name}")
        print('='*60)
        
        try:
            benchmark = HFCLIPObjectSizeBenchmark(model_name=model_name)
            results[model_name] = benchmark.benchmark_dataset(csv_path)
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    
    return results

def analyze_size_performance_hf(csv_path, size_categories=['tiny', 'small', 'medium', 'large', 'huge']):
    """Detailed analysis of performance across size categories using HF CLIP"""
    benchmark = HFCLIPObjectSizeBenchmark()
    results_df = benchmark.benchmark_dataset(csv_path)
    
    print("\nDETAILED SIZE ANALYSIS (HUGGING FACE CLIP):")
    print("-" * 60)
    
    for size in size_categories:
        size_data = results_df[results_df['size_category'] == size]
        if len(size_data) > 0:
            accuracy = size_data['is_correct'].mean()
            top5_accuracy = size_data['top5_correct'].mean()
            avg_confidence = size_data['confidence'].mean()
            count = len(size_data)
            
            print(f"{size.upper():>8}: {accuracy:.1%} top1 | {top5_accuracy:.1%} top5 | {avg_confidence:.3f} conf | {count:>4} objects")

if __name__ == "__main__":
    main()