# MODIFIED: encoder.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
import copy

BASE_DIR = "./QEncoder_SP500_prediction/encoder_details/"


def construct_autoencoder_circuit(args, weights, features=None):
    dev = qml.device("default.qubit", wires=args.num_latent + 2 * args.num_trash + 1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def autoencoder_circuit(weights, features=None):
        n_qubits = args.num_latent + args.num_trash
        try:
            weights = weights.reshape(args.depth, n_qubits)
        except RuntimeError as e:
            raise ValueError(f"Weight reshaping error: {e}, Expected shape: ({args.depth}, {n_qubits})")

        if features is not None:
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="X")

        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

        aux_qubit = args.num_latent + 2 * args.num_trash
        qml.Hadamard(wires=aux_qubit)
        for i in range(args.num_trash):
            qml.CSWAP(wires=[aux_qubit, args.num_latent + i, args.num_latent + args.num_trash + i])
        qml.Hadamard(wires=aux_qubit)

        return qml.probs(wires=[aux_qubit])

    return autoencoder_circuit(weights, features)


class AutoEncoder(nn.Module):
    def __init__(self, args, weights=None):
        super().__init__()
        self.n_qubits = args.num_latent + args.num_trash
        self.args = args
        self.weights = (
            weights if weights is not None else
            nn.Parameter(0.01 * torch.rand(args.depth, self.n_qubits), requires_grad=True)
        )

    def forward(self, features):
        probs = construct_autoencoder_circuit(self.args, self.weights, features)
        # Return probability of measuring 1, which corresponds to information loss
        return probs[:, 1]


def autoencoder_circuit_trained(weights, args):
    n_qubits = args.num_latent + args.num_trash
    weights = weights.reshape(args.depth, n_qubits)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))


def train_encoder(flattened, args):
    print("\nStarting Encoder Training...", flush=True)

    # Create directory if it doesn't exist
    os.makedirs(BASE_DIR, exist_ok=True)

    best_model_path = os.path.join(BASE_DIR, f"{args.dataset}_best_encoder.pth")

    if os.path.exists(best_model_path):
        print("Best encoder model already exists. Loading it.")
        enc = AutoEncoder(args)
        enc.load_state_dict(torch.load(best_model_path))
        enc.eval()
        return enc

    enc = AutoEncoder(args)
    # MODIFIED: Optimizer changed to RMSprop for consistency and better performance 
    opt = optim.RMSprop(enc.parameters(), lr=args.lr)

    best_loss = float("inf")

    # MODIFIED: Training iterations are now controlled by the new argument 
    for i in range(1, args.encoder_train_iter + 1):
        train_indices = np.random.randint(0, len(flattened), (args.batch_size,))
        features = torch.tensor(flattened[train_indices], dtype=torch.float32)

        opt.zero_grad()
        out = enc(features)
        loss = torch.sum(out) / args.batch_size
        loss.backward()
        opt.step()

        current_loss = loss.item()

        if i % 10 == 0:
            print(f"Encoder Iteration: {i} | Loss: {current_loss:.4f}", flush=True)

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(enc.state_dict(), best_model_path)
            print(f"--> New BEST encoder model saved at iteration {i} with loss {best_loss:.4f}")

    print(f"\nEncoder training finished. Best model saved at '{best_model_path}' with loss {best_loss:.4f}\n")
    return enc