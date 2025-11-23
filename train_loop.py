# MODIFIED: train_loop.py

import torch
import torch.optim as optim
import os
from sklearn.metrics import r2_score, mean_squared_error
from pennylane import numpy as np


def accuracy(y, y_hat):
    """Calculates R2 and MSE metrics."""
    # Detach tensors from the computation graph and move to CPU
    y = y.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()

    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    return r2, mse


def train(
        model,
        train_set,
        labels_train,
        validation_set,
        labels_val,
        args,
):
    print("Training Started...", flush=True)

    opt = optim.RMSprop(model.parameters(), lr=args.lr)

    # Define directories for saving results
    base_dir = "./QEncoder_SP500_prediction/evaluation_results/"
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    experiment_name = f"{args.dataset}_{args.loss}_{args.depth}"
    latest_checkpoint_path = os.path.join(weights_dir, f"{experiment_name}_latest_model.pth")
    best_checkpoint_path = os.path.join(weights_dir, f"{experiment_name}_best_model.pth")

    start_iter = 0
    best_val_mse = float('inf')
    history = {'losses': [], 'val_r2': [], 'val_mse': []}

    # MODIFIED: Added checkpoint loading to resume training
    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration'] + 1
        best_val_mse = checkpoint['best_val_mse']
        history = checkpoint['history']
        print(f"Resumed from iteration {start_iter}.")

    if args.loss == "MSE":
        loss_fun = torch.nn.MSELoss()
    else:
        # Note: BCE loss is not suitable for this regression task. Sticking to MSE.
        print("Warning: Only MSE loss is recommended for this regression task.", flush=True)
        loss_fun = torch.nn.MSELoss()

    for i in range(start_iter, args.train_iter):
        model.train()  # Set model to training mode

        # MODIFIED: Vectorized batch creation for performance
        # Create a single tensor for the batch instead of a list of samples.
        train_indices = np.random.randint(0, len(train_set), (args.batch_size,))
        features = torch.tensor(train_set[train_indices], dtype=torch.float32)
        train_labels = torch.tensor(labels_train[train_indices], dtype=torch.float32)

        opt.zero_grad()
        out = model(features)
        loss = loss_fun(out, train_labels)
        loss.backward()
        opt.step()

        history['losses'].append(loss.item())

        # Validation and Checkpointing
        if i % args.eval_every == 0:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # MODIFIED: Vectorized validation batch creation
                val_indices = np.random.randint(0, len(validation_set), (args.batch_size,))
                val_features = torch.tensor(validation_set[val_indices], dtype=torch.float32)
                val_labels = torch.tensor(labels_val[val_indices], dtype=torch.float32)

                out_val = model(val_features)
                r2_val, mse_val = accuracy(val_labels, out_val)

                history['val_r2'].append(r2_val)
                history['val_mse'].append(mse_val)

                print(f"Iter: {i} | Train Loss: {loss.item():.4f} | Val R2: {r2_val:.4f} | Val MSE: {mse_val:.4f}",
                      flush=True)

                # MODIFIED: Improved checkpoint saving logic
                # Save the "best" model based on validation MSE
                if mse_val < best_val_mse:
                    best_val_mse = mse_val
                    torch.save({'model_state_dict': model.state_dict()}, best_checkpoint_path)
                    print(f"--> New BEST model saved with Val MSE: {best_val_mse:.4f}")

        # Save the "latest" model checkpoint for resuming
        torch.save({
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_val_mse': best_val_mse,
            'history': history,
        }, latest_checkpoint_path)

    print("Training finished.", flush=True)
    return model