import json
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

class GenerativeVLMBenchmark:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device=None):
        """
        Initialize VLM benchmark with different model options:
        - "Salesforce/blip-vqa-base" - BLIP VQA model
        - "Salesforce/blip-vqa-capfilt-large" - Larger BLIP model  
        - "microsoft/git-base-vqav2" - GIT model
        - "dandelin/vilt-b32-finetuned-vqa" - ViLT model
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading VLM: {model_name} on {self.device}")
        
        # Load model and processor based on model type
        if "blip" in model_name.lower():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        else:
            # For other models like GIT, ViLT, etc.
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def answer_vqa_question(self, image_path, question):
        """
        Get VLM answer to a VQA question
        
        Args:
            image_path: Path to image
            question: Question string
            
        Returns:
            Generated answer string
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs
            if "blip" in self.model_name.lower():
                inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=20, num_beams=5)
                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                # For other models (adjust as needed)
                inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=20, num_beams=5)
                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return answer.strip()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return "error"
    
    def normalize_answer(self, answer):
        """Normalize VLM answers to yes/no for evaluation"""
        if answer is None or answer == "error":
            return 'unknown'
        
        answer_lower = str(answer).lower().strip()
        
        # Remove common prefixes that VLMs add
        answer_lower = re.sub(r'^(the answer is|answer:|yes,|no,)\s*', '', answer_lower)
        
        # Yes variations
        yes_patterns = ['yes', 'true', 'correct', 'present', 'visible', 'there is', 'i can see', 
                       'appears', 'shown', 'displayed', 'contains', 'includes']
        if any(pattern in answer_lower for pattern in yes_patterns):
            return 'yes'
        
        # No variations
        no_patterns = ['no', 'false', 'incorrect', 'absent', 'not', 'cannot', 'can\'t', 
                      'unable', 'don\'t see', 'not visible', 'not present']
        if any(pattern in answer_lower for pattern in no_patterns):
            return 'no'
        
        # If answer is very short and doesn't match patterns, try exact matching
        if len(answer_lower.split()) == 1:
            if answer_lower in ['yes', 'y', '1', 'true']:
                return 'yes'
            elif answer_lower in ['no', 'n', '0', 'false']:
                return 'no'
        
        return 'unknown'
    
    def evaluate_vqa_dataset(self, vqa_file_path, output_dir="/data/azfarm/siddhant/ICCV/vlm_benchmark_results"):
        """
        Evaluate VLM on VQA dataset using image paths from JSON
        
        Args:
            vqa_file_path: Path to VQA JSON file
            output_dir: Output directory for results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load VQA questions
        print(f"Loading VQA questions from {vqa_file_path}")
        with open(vqa_file_path, 'r') as f:
            questions = json.load(f)
        
        print(f"Found {len(questions)} questions")
        
        results = []
        correct = 0
        missing_images = 0
        
        # Process each question
        print("Running VLM evaluation...")
        for i, q in enumerate(tqdm(questions)):
            image_id = q['image_id']
            question = q['question']
            ground_truth = q['answer']
            target_class = q['target_class']
            target_size = q.get('target_size', 'unknown')
            dataset = q.get('dataset', 'unknown')
            
            # Get image path directly from JSON
            image_path = q['image_path']
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                missing_images += 1
                if missing_images <= 10:  # Show first 10 missing
                    print(f"Warning: Image not found: {image_path}")
                elif missing_images == 11:
                    print("... (suppressing further missing image warnings)")
                continue
            
            # Get VLM prediction
            predicted_answer = self.answer_vqa_question(image_path, question)
            
            # Normalize answers
            pred_normalized = self.normalize_answer(predicted_answer)
            gt_normalized = ground_truth.lower()
            
            is_correct = (pred_normalized == gt_normalized)
            if is_correct:
                correct += 1
            
            results.append({
                'image_id': image_id,
                'image_path': image_path,
                'dataset': dataset,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_raw': predicted_answer,
                'predicted_normalized': pred_normalized,
                'correct': is_correct,
                'target_class': target_class,
                'target_size': target_size,
                'question_type': q.get('question_type', 'unknown')
            })
            
            # Print progress every 100 questions
            if (i + 1) % 100 == 0:
                current_acc = correct / len(results) if results else 0
                processed_count = len(results)
                print(f"Progress: {i+1}/{len(questions)} | Processed: {processed_count} | Current accuracy: {current_acc:.3f}")
        
        print(f"\n✅ Evaluation complete!")
        print(f"📊 Successfully processed: {len(results)} questions")
        print(f"⚠️  Missing images: {missing_images}")
        
        if not results:
            print("❌ No valid results generated. Please check image paths in JSON.")
            return None
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        
        results_df = pd.DataFrame(results)
        detailed_path = os.path.join(output_dir, f"detailed_results_{model_safe_name}_{timestamp}.csv")
        results_df.to_csv(detailed_path, index=False)
        
        # Generate analysis
        self._analyze_results(results_df, output_dir, timestamp)
        
        return results_df
    
    def _analyze_results(self, results_df, output_dir, timestamp):
        """Generate comprehensive analysis"""
        
        print("\n" + "="*80)
        print(f"VLM BENCHMARK RESULTS - {self.model_name}")
        print("="*80)
        
        # Overall metrics
        total_questions = len(results_df)
        correct_answers = results_df['correct'].sum()
        overall_accuracy = correct_answers / total_questions
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Model: {self.model_name}")
        print(f"Total questions: {total_questions}")
        print(f"Correct answers: {correct_answers}")
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # Filter out unknown predictions for cleaner analysis
        valid_results = results_df[results_df['predicted_normalized'] != 'unknown']
        if len(valid_results) < len(results_df):
            unknown_count = len(results_df) - len(valid_results)
            print(f"Unknown/unparseable answers: {unknown_count} ({unknown_count/total_questions*100:.1f}%)")
        
        # Performance by size category
        print(f"\n🎯 PERFORMANCE BY OBJECT SIZE:")
        print("-" * 60)
        
        size_analysis = results_df.groupby('target_size').agg({
            'correct': ['count', 'sum', 'mean']
        }).round(3)
        size_analysis.columns = ['Total_Questions', 'Correct_Answers', 'Accuracy']
        
        # Order by size
        size_order = ['tiny', 'small', 'medium', 'large', 'huge']
        size_analysis = size_analysis.reindex([s for s in size_order if s in size_analysis.index])
        
        print(size_analysis)
        
        print(f"\n📊 SIZE PERFORMANCE SUMMARY:")
        for size in size_order:
            if size in size_analysis.index:
                acc = size_analysis.loc[size, 'Accuracy']
                count = size_analysis.loc[size, 'Total_Questions']
                print(f"  {size.upper():>6}: {acc:.1%} accuracy ({count:>3} questions)")
        
        # Performance by answer type
        print(f"\nPERFORMANCE BY ANSWER TYPE:")
        answer_analysis = results_df.groupby('ground_truth').agg({
            'correct': ['count', 'sum', 'mean']
        }).round(3)
        answer_analysis.columns = ['Total_Questions', 'Correct_Answers', 'Accuracy']
        print(answer_analysis)
        
        # Performance by class
        print(f"\nTOP 10 CLASSES BY ACCURACY:")
        class_analysis = results_df.groupby('target_class').agg({
            'correct': ['count', 'mean']
        }).round(3)
        class_analysis.columns = ['Count', 'Accuracy']
        class_analysis = class_analysis[class_analysis['Count'] >= 5]  # At least 5 questions
        top_classes = class_analysis.sort_values('Accuracy', ascending=False).head(10)
        print(top_classes)
        
        print(f"\nWORST 10 CLASSES BY ACCURACY:")
        worst_classes = class_analysis.sort_values('Accuracy', ascending=True).head(10)
        print(worst_classes)
        
        # Confusion analysis
        print(f"\nCONFUSION ANALYSIS:")
        confusion = pd.crosstab(results_df['ground_truth'], results_df['predicted_normalized'], 
                               margins=True, normalize='index')
        print(confusion.round(3))
        
        # Save summary
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'total_questions': int(total_questions),
            'correct_answers': int(correct_answers),
            'overall_accuracy': float(overall_accuracy),
            'size_performance': size_analysis.to_dict(),
            'answer_type_performance': answer_analysis.to_dict(),
            'confusion_matrix': confusion.to_dict()
        }
        
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        summary_path = os.path.join(output_dir, f"summary_{model_safe_name}_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create visualizations
        self._create_visualizations(results_df, output_dir, timestamp)
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"Summary: {summary_path}")
    
    def _create_visualizations(self, results_df, output_dir, timestamp):
        """Create visualization plots"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'VLM Performance Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Size Category
        ax1 = axes[0, 0]
        size_acc = results_df.groupby('target_size')['correct'].mean()
        size_counts = results_df.groupby('target_size').size()
        
        # Order by size
        size_order = ['tiny', 'small', 'medium', 'large', 'huge']
        size_acc_ordered = [size_acc.get(size, 0) for size in size_order if size in size_acc.index]
        size_counts_ordered = [size_counts.get(size, 0) for size in size_order if size in size_counts.index]
        size_labels = [size for size in size_order if size in size_acc.index]
        
        colors = ['red', 'orange', 'gold', 'green', 'blue'][:len(size_acc_ordered)]
        bars = ax1.bar(range(len(size_acc_ordered)), size_acc_ordered, color=colors)
        
        ax1.set_xlabel('Object Size Category')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('🎯 Accuracy by Object Size')
        ax1.set_xticks(range(len(size_labels)))
        ax1.set_xticklabels(size_labels)
        ax1.set_ylim(0, 1)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, size_counts_ordered)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 2. Answer Type Performance
        ax2 = axes[0, 1]
        answer_acc = results_df.groupby('ground_truth')['correct'].mean()
        answer_counts = results_df.groupby('ground_truth').size()
        
        bars = ax2.bar(answer_acc.index, answer_acc.values, color=['lightcoral', 'lightblue'])
        ax2.set_xlabel('Ground Truth Answer')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Performance by Answer Type')
        ax2.set_ylim(0, 1)
        
        for i, (bar, count) in enumerate(zip(bars, answer_counts.values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # 3. Top Classes Performance
        ax3 = axes[1, 0]
        class_acc = results_df.groupby('target_class')['correct'].mean()
        class_counts = results_df.groupby('target_class').size()
        
        # Filter classes with at least 5 questions and show top 10
        valid_classes = class_acc[class_counts >= 5].sort_values(ascending=True).tail(10)
        
        ax3.barh(range(len(valid_classes)), valid_classes.values)
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Object Class')
        ax3.set_title('Top 10 Classes by Accuracy')
        ax3.set_yticks(range(len(valid_classes)))
        ax3.set_yticklabels(valid_classes.index, fontsize=9)
        ax3.set_xlim(0, 1)
        
        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        pred_dist = results_df['predicted_normalized'].value_counts()
        
        colors = ['lightgreen' if x == 'yes' else 'lightcoral' if x == 'no' else 'lightgray' 
                 for x in pred_dist.index]
        bars = ax4.bar(pred_dist.index, pred_dist.values, color=colors)
        ax4.set_xlabel('Predicted Answer')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of VLM Predictions')
        
        # Add percentages
        total = pred_dist.sum()
        for bar, count in zip(bars, pred_dist.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{count/total*100:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        plot_path = os.path.join(output_dir, f"analysis_plots_{model_safe_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: {plot_path}")

def main():
    """Main benchmarking function"""
    
    # Configuration
    MODEL_NAME = "Salesforce/blip-vqa-base"
    VQA_FILE = "/data/azfarm/siddhant/ICCV/simple_vqa_dataset/simple_vqa_questions.json"
    OUTPUT_DIR = "/data/azfarm/siddhant/ICCV/vlm_benchmark_results"
    
    # Check if files exist
    if not os.path.exists(VQA_FILE):
        print(f"❌ VQA file not found: {VQA_FILE}")
        return
    
    # Initialize benchmark
    print("🚀 Starting VLM Benchmark...")
    benchmark = GenerativeVLMBenchmark(model_name=MODEL_NAME)
    
    # Run evaluation
    results_df = benchmark.evaluate_vqa_dataset(VQA_FILE, OUTPUT_DIR)
    
    print(f"\n✅ Benchmarking complete!")
    
    return results_df

def benchmark_multiple_models():
    """Benchmark multiple VLM models for comparison"""
    
    models = [
        "Salesforce/blip-vqa-base",
        "Salesforce/blip-vqa-capfilt-large",
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model_name}")
        print('='*60)
        
        try:
            benchmark = GenerativeVLMBenchmark(model_name=model_name)
            results[model_name] = benchmark.evaluate_vqa_dataset(
                "/data/azfarm/siddhant/ICCV/simple_vqa_dataset/simple_vqa_questions.json"
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    return results

if __name__ == "__main__":
    main()