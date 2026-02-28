#!/usr/bin/env python3
"""
Generate a simple PyTorch model for testing Boat PyTorch loader.
"""

import torch
import torch.nn as nn

def create_simple_model():
    """Create a simple sequential model."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
        nn.Softmax(dim=1)
    )
    return model

def main():
    print("Generating test PyTorch model...")

    # Create model
    model = create_simple_model()

    # Save as TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save("test_simple_model.pt")

    # Also save state dict for reference
    torch.save(model.state_dict(), "test_simple_model_state_dict.pth")

    print("Saved test_simple_model.pt (TorchScript)")
    print("Saved test_simple_model_state_dict.pth (state dict)")

    # Print model info
    print("\nModel architecture:")
    print(model)

    print("\nModel parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} ({param.dtype})")
        total_params += param.numel()

    print(f"\nTotal parameters: {total_params}")

if __name__ == "__main__":
    main()