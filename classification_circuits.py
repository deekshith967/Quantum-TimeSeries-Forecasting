# MODIFIED: classification_circuits.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pennylane as qml
import torch


def circuit_7(weights, args):
    """
    A specific variational quantum circuit layer structure.
    """
    n_qubits = args.num_latent + args.num_trash
    depth = args.depth

    # MODIFIED: This formula now correctly matches the one in classification_model.py
    weights_per_layer = 4 * n_qubits + (n_qubits - 1)

    try:
        weights = weights.reshape(depth, weights_per_layer)
    except RuntimeError as e:
        # This error handling will now catch any future mismatches
        raise ValueError(f"Weight reshaping error for circuit_7: {e}")

    for j in range(depth):
        layer_weights = weights[j]
        w_count = 0
        # This loop structure correctly uses 49 weights per layer for 10 qubits
        for i in range(n_qubits):
            qml.RX(layer_weights[w_count], wires=i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(layer_weights[w_count], wires=i)
            w_count += 1
        for i in range(0, n_qubits - 1, 2):
            qml.CRZ(layer_weights[w_count], wires=[i + 1, i])
            w_count += 1
        for i in range(n_qubits):
            qml.RX(layer_weights[w_count], wires=i)
            w_count += 1
        for i in range(n_qubits):
            qml.RZ(layer_weights[w_count], wires=i)
            w_count += 1
        for i in range(1, n_qubits - 1, 2):
            qml.CRZ(layer_weights[w_count], wires=[i + 1, i])
            w_count += 1


def construct_classification_circuit(args, weights, features):
    """
    MODIFIED: This circuit is now simplified to align with the results paper.
    It performs direct angle embedding of the 10 features from the 2-day window,
    followed by the variational processing circuit. The complex EIP loop is removed.
    """
    n_qubits = args.num_latent + args.num_trash
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def classification_circuit(model_weights, features):
        # Embed the 10 features from the 2-day window onto the 10 qubits
        qml.AngleEmbedding(features, wires=range(n_qubits), rotation="X")

        # Apply the trainable variational layers
        circuit_7(model_weights, args)

        # MODIFIED: Return the expectation value of the first qubit for regression
        return qml.expval(qml.PauliZ(0))

    return classification_circuit(weights, features)