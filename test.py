# MODIFIED: test.py

import os
import torch
from .metrics import metrics


def test(
        model,
        args,
        test_dir,
        test_set,
        labels_test,
):
    # MODIFIED: Path now points specifically to the best model saved during training
    experiment_name = f"{args.dataset}_{args.loss}_{args.depth}"
    best_model_path = os.path.join(test_dir, f"{experiment_name}_best_model.pth")

    print(f"\n--- Starting Testing ---")
    if not os.path.exists(best_model_path):
        print(f"Error: No best model found at {best_model_path}. Please train the model first.")
        return

    print(f"Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded. Running inference on the test set...")
    with torch.no_grad():
        # Ensure test set and labels are tensors
        test_features = torch.tensor(test_set, dtype=torch.float32)
        test_labels = torch.tensor(labels_test, dtype=torch.float32)

        test_out = model(test_features)

    print("\n--- Test Results ---")
    metrics(test_out.cpu().numpy(), test_labels.cpu().numpy())
    print("--- Testing Finished ---\n")