# MODIFIED: classification_model.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from .classification_circuits import construct_classification_circuit


class Classifier(nn.Module):
    def __init__(self, encoder, args, model_weights=None):
        super().__init__()
        self.args = args
        self.encoder = encoder

        n_qubits = args.num_latent + args.num_trash

        # MODIFIED: Corrected the weight calculation formula.
        # The original formula (4 * n_qubits - 2) was incorrect for the circuit structure.
        # The circuit_7 function actually requires (4 * n_qubits + n_qubits - 1) weights per layer.
        n_weights_per_layer = 4 * n_qubits + (n_qubits - 1)
        n_weights = n_weights_per_layer * args.depth

        self.model_weights = (
            nn.Parameter(model_weights)
            if model_weights is not None
            else nn.Parameter(0.1 * torch.rand(n_weights), requires_grad=True)
        )

    def forward(self, features):
        """
        MODIFIED: The forward pass is now vectorized.
        The per-sample 'for' loop has been removed for a massive performance gain.
        The QNode now processes the entire batch of features at once.
        """
        raw_predictions = construct_classification_circuit(
            self.args, self.model_weights, features
        )

        # Scale from [-1, 1] to [0, 1]
        scaled_predictions = (raw_predictions + 1) / 2.0

        return scaled_predictions.to(torch.float32)