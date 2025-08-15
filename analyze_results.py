#!/usr/bin/env python3
"""
Analysis script for multi-video inference results
Provides statistics and sample comparisons between ground truth and model predictions.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze multi-video inference results")
    parser.add_argument(
        "--results_path", 
        type=str, 
        required=True,
        help="Path to inference results JSON file"
    )
    parser.add_argument(
        "--output_summary", 
        type=str, 
        default=None,
        help="Path to save analysis summary (optional)"
    )
    parser.add_argument(
        "--show_examples", 
        type=int, 
        default=5,
        help="Number of examples to show"
    )
    parser.add_argument(
        "--show_errors", 
        action="store_true",
        help="Show error details"
    )
    
    return parser.parse_args()


def calculate_basic_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic statistics about the results"""
    if not results:
        return {}
    
    # Response lengths
    model_lengths = [len(r["model_response"]) for r in results]
    gt_lengths = [len(r["ground_truth"]) for r in results]
    
    # Video counts
    video_counts = [r.get("video_count", 0) for r in results]
    
    stats = {
        "total_samples": len(results),
        "avg_model_response_length": sum(model_lengths) / len(model_lengths),
        "avg_ground_truth_length": sum(gt_lengths) / len(gt_lengths),
        "min_model_response_length": min(model_lengths),
        "max_model_response_length": max(model_lengths),
        "min_ground_truth_length": min(gt_lengths),
        "max_ground_truth_length": max(gt_lengths),
        "avg_video_count": sum(video_counts) / len(video_counts),
        "min_video_count": min(video_counts),
        "max_video_count": max(video_counts),
        "video_count_distribution": {}
    }
    
    # Video count distribution
    from collections import Counter
    video_dist = Counter(video_counts)
    stats["video_count_distribution"] = dict(sorted(video_dist.items()))
    
    return stats


def calculate_text_similarity_basic(text1: str, text2: str) -> float:
    """Basic text similarity using word overlap (simple metric)"""
    # Simple word-level Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def analyze_content_types(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze types of questions and responses"""
    question_types = {
        "descriptive": 0,  # "describe", "what do you see"
        "analytical": 0,   # "analyze", "compare", "patterns"
        "counting": 0,     # "how many", "count"
        "temporal": 0,     # "sequence", "order", "timeline"
        "other": 0
    }
    
    for result in results:
        question = result["user_question"].lower()
        
        if any(word in question for word in ["describe", "what do you see", "what is", "show"]):
            question_types["descriptive"] += 1
        elif any(word in question for word in ["analyze", "compare", "pattern", "theme", "summary"]):
            question_types["analytical"] += 1
        elif any(word in question for word in ["how many", "count", "number of"]):
            question_types["counting"] += 1
        elif any(word in question for word in ["sequence", "order", "timeline", "first", "then", "next"]):
            question_types["temporal"] += 1
        else:
            question_types["other"] += 1
    
    return question_types


def analyze_results(results_path: str, args):
    """Main analysis function"""
    print(f"Loading results from: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    results = data.get("results", [])
    errors = data.get("errors", [])
    
    print("\n=== INFERENCE RESULTS ANALYSIS ===")
    print(f"Model: {metadata.get('model_path', 'Unknown')}")
    print(f"Test Dataset: {metadata.get('test_data_path', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print()
    
    # Basic statistics
    print("=== BASIC STATISTICS ===")
    if results:
        stats = calculate_basic_stats(results)
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Average Model Response Length: {stats['avg_model_response_length']:.1f} chars")
        print(f"Average Ground Truth Length: {stats['avg_ground_truth_length']:.1f} chars")
        print(f"Average Videos per Sample: {stats['avg_video_count']:.1f}")
        print(f"Video Count Range: {stats['min_video_count']} - {stats['max_video_count']}")
        
        print("\nVideo Count Distribution:")
        for count, frequency in stats['video_count_distribution'].items():
            print(f"  {count} videos: {frequency} samples")
    else:
        print("No successful results to analyze.")
    
    # Error analysis
    if errors:
        print(f"\n=== ERRORS ({len(errors)}) ===")
        if args.show_errors:
            for error in errors[:5]:  # Show first 5 errors
                print(f"Sample {error.get('id', 'Unknown')}: {error.get('error', 'Unknown error')}")
            if len(errors) > 5:
                print(f"... and {len(errors) - 5} more errors")
        else:
            print(f"Use --show_errors to see error details")
    
    # Content type analysis
    if results:
        print("\n=== QUESTION TYPE ANALYSIS ===")
        question_types = analyze_content_types(results)
        for qtype, count in question_types.items():
            percentage = (count / len(results)) * 100
            print(f"{qtype.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Similarity analysis (basic)
    if results:
        print("\n=== RESPONSE SIMILARITY ANALYSIS ===")
        similarities = []
        for result in results:
            sim = calculate_text_similarity_basic(
                result["model_response"], 
                result["ground_truth"]
            )
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        print(f"Average Word Overlap Similarity: {avg_similarity:.3f}")
        
        # Distribution
        high_sim = sum(1 for s in similarities if s > 0.5)
        med_sim = sum(1 for s in similarities if 0.2 <= s <= 0.5)
        low_sim = sum(1 for s in similarities if s < 0.2)
        
        print(f"High similarity (>0.5): {high_sim} samples")
        print(f"Medium similarity (0.2-0.5): {med_sim} samples")  
        print(f"Low similarity (<0.2): {low_sim} samples")
    
    # Example outputs
    if results and args.show_examples > 0:
        print(f"\n=== SAMPLE RESULTS ({min(args.show_examples, len(results))}) ===")
        
        for i, result in enumerate(results[:args.show_examples]):
            print(f"\n--- Sample {i+1}: {result['id']} ---")
            print(f"Videos: {result.get('video_count', 0)}")
            print(f"Question: {result['user_question'][:150]}{'...' if len(result['user_question']) > 150 else ''}")
            print(f"\nGround Truth: {result['ground_truth'][:200]}{'...' if len(result['ground_truth']) > 200 else ''}")
            print(f"\nModel Response: {result['model_response'][:200]}{'...' if len(result['model_response']) > 200 else ''}")
            
            # Basic similarity for this sample
            sim = calculate_text_similarity_basic(result['model_response'], result['ground_truth'])
            print(f"\nWord Overlap Similarity: {sim:.3f}")
    
    # Save summary if requested
    if args.output_summary:
        summary = {
            "metadata": metadata,
            "statistics": stats if results else {},
            "question_types": question_types if results else {},
            "average_similarity": avg_similarity if results else 0,
            "error_count": len(errors),
            "analysis_timestamp": data.get("metadata", {}).get("timestamp", "Unknown")
        }
        
        with open(args.output_summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary saved to: {args.output_summary}")


if __name__ == "__main__":
    args = parse_args()
    analyze_results(args.results_path, args)