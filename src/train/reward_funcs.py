import os
import re
import math
from datetime import datetime
from math_verify import parse, verify

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def numerical_video_reward(completions, assistant, **kwargs):
    """
    Custom reward function for numerical answers in video analysis.
    
    Rewards structure:
    - If ground truth has a number: RMSE-based reward (1.0 for exact, decreases with error)
    - If ground truth has no number: neutral reward (0.5)
    - If prediction has number when GT has number: base reward (0.1) + RMSE bonus
    - Range emphasis: Numbers far from 50 get higher weight for correct answers
    """
    contents = [completion[0]["content"] for completion in completions]
    solutions = [a['content'] for a in assistant]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    def extract_number(text):
        """Extract numerical value from text, handling percentages."""
        # Remove common formatting and extract numbers
        text = text.strip().lower()
        
        # Handle percentages (e.g., "65.4%" -> 65.4)
        percentage_match = re.search(r'(\d+(?:\.\d+)?)%', text)
        if percentage_match:
            return float(percentage_match.group(1))
        
        # Handle regular numbers (with optional decimal)
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if number_match:
            return float(number_match.group(1))
        
        return None
    
    def calculate_rmse_reward(predicted, ground_truth, range_min=20, range_max=80, center=50):
        """
        Calculate RMSE-based reward with emphasis on far-from-center values.
        
        Args:
            predicted: Predicted numerical value
            ground_truth: True numerical value  
            range_min, range_max: Expected range of values
            center: Center value for range emphasis
        """
        if predicted is None:
            return 0.0
        
        # Base RMSE calculation
        error = abs(predicted - ground_truth)
        max_error = range_max - range_min  # Maximum possible error in range
        
        # RMSE-style reward: starts at 1.0 for exact match, decreases quadratically
        rmse_reward = max(0.0, 1.0 - (error / max_error) ** 2)
        
        # Range emphasis: Give bonus for correct answers far from center
        distance_from_center = abs(ground_truth - center)
        max_distance = max(range_max - center, center - range_min)
        range_bonus = 1.0 + (distance_from_center / max_distance) * 0.5  # Up to 50% bonus
        
        return rmse_reward * range_bonus
    
    for content, solution in zip(contents, solutions):
        # Extract numbers from ground truth and prediction
        gt_number = extract_number(solution)
        pred_number = extract_number(content)
        
        if gt_number is None:
            # No number in ground truth -> neutral reward
            reward = 0.5
        else:
            # Ground truth has a number
            if pred_number is None:
                # Prediction has no number -> very low reward
                reward = 0.0
            else:
                # Both have numbers -> calculate RMSE-based reward
                base_reward = 0.1  # Base reward for having a number
                rmse_reward = calculate_rmse_reward(pred_number, gt_number)
                reward = base_reward + rmse_reward * 0.9  # Scale RMSE part to [0.1, 1.0]
        
        rewards.append(min(1.0, max(0.0, reward)))  # Clamp to [0, 1]
        
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_rewards.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Numerical Video Reward: {reward:.3f} -------------\n")
                f.write(f"Ground Truth: {solution} (number: {gt_number})\n")
                f.write(f"Prediction: {content} (number: {pred_number})\n")
                if gt_number is not None and pred_number is not None:
                    f.write(f"Error: {abs(pred_number - gt_number):.2f}\n")
                f.write("\n")
    
    return rewards
