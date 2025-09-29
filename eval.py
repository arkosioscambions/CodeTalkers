import pandas as pd
import argparse

def evaluate(pred_path, gt_path):
    # Load model outputs
    df = pd.read_json(pred_path, lines=True)
    
    # Load ground truth labels
    df2 = pd.read_csv(gt_path)
    ground_truth = df2["ground_truth"].astype(str).str.strip()
    
    # Extract model completions
    model_responses = df["completion"].astype(str).str.strip()
    
    # Ensure lengths match
    assert len(ground_truth) == len(model_responses), "Mismatch in number of samples!"
    
    # Compute Exact Match (EM)
    correct = sum(gt == pred for gt, pred in zip(ground_truth, model_responses))
    em = correct / len(ground_truth)
    
    print(f"Exact Match Score: {em:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Exact Match between model completions and ground truth.")
    parser.add_argument("--pred", type=str, required=True, help="Path to model predictions JSONL file")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth CSV file")
    args = parser.parse_args()

    evaluate(args.pred, args.gt)
