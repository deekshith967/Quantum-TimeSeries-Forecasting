# MODIFIED: main.py

import os
import argparse
from QEncoder_SP500_prediction.train_loop import train
from .encoder import train_encoder
from .classification_model import Classifier
from .dataset import load_dataset, split_features_labels
from .test import test

parser = argparse.ArgumentParser()
parser.add_argument("--loss", dest="loss", type=str, default="MSE", help="Loss function (MSE or BCE)")
parser.add_argument("--eval_every", dest="eval_every", type=int, default=10, help="Evaluate on validation set every N iterations")
parser.add_argument("--dataset", dest="dataset", type=str, default="sp500", help="Dataset to use: sp500, nifty, or wti")
# MODIFIED: Added encoder_train_iter to control encoder training separately 
parser.add_argument("--encoder_train_iter", dest="encoder_train_iter", type=int, default=300, help="Number of training iterations for the encoder")
parser.add_argument("--train_iter", dest="train_iter", type=int, default=200, help="Number of training iterations for the classifier")
parser.add_argument("--depth", dest="depth", type=int, default=2, help="Depth of the variational circuit")
parser.add_argument("--mode", dest="mode", type=str, default="train", help="Mode: train or test")
# MODIFIED: Default qubit counts adjusted to 10 total for the 2-day model 
parser.add_argument("--num_latent", dest="num_latent", type=int, default=4, help="Number of latent qubits")
parser.add_argument("--num_trash", dest="num_trash", type=int, default=6, help="Number of trash qubits")
parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument("--bs", dest="batch_size", type=int, default=256, help="Batch size for training")
args = parser.parse_args()

print(f"{os.getpid()=}")
print(f"Running in mode: {args.mode}")
print(f"Dataset: {args.dataset}")
print(f"Total qubits for encoder: {args.num_latent + args.num_trash}")

# Load dataset
X, Y, tX, tY, flattened = load_dataset(args)
print(f"Training data shape: {X.shape}")
print(f"Test data shape: {tX.shape}")

# Train the encoder
trained_encoder = train_encoder(flattened, args)

# Initialize the classifier model
model = Classifier(trained_encoder, args)

# Split the data into training and validation sets
train_set, validation_set, labels_train, labels_val = split_features_labels(X, Y, 0.2)
test_set, labels_test = tX, tY

# Perform training or testing based on the mode
if args.mode == "train":
    train(
        model,
        train_set,
        labels_train,
        validation_set,
        labels_val,
        args,
    )
elif args.mode == "test":
    BASE_DIR = "./QEncoder_SP500_prediction/"
    test_dir = os.path.join(BASE_DIR, 'evaluation_results/weights/')
    test(
        model,
        args,
        test_dir,
        test_set,
        labels_test
    )