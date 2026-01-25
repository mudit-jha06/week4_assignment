#!/usr/bin/env python3
"""
Convert PyTorch models (.pth) to ONNX format (.onnx)

Usage:
    python convert_to_onnx.py
"""

import torch
import torch.nn as nn
from pathlib import Path


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.
    Must match the architecture used during training.
    """
    
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # After 4 pooling layers: 150 -> 75 -> 37 -> 18 -> 9
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Conv block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def convert_pth_to_onnx(pth_path, onnx_path, num_classes=6):
    """Convert a .pth model to .onnx format."""
    print(f"Converting {pth_path} -> {onnx_path}")
    
    # Load model
    model = SimpleCNN(num_classes=num_classes)
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input (batch=1, channels=3, height=150, width=150)
    dummy_input = torch.randn(1, 3, 150, 150)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11
    )
    
    print(f"  ✓ Saved to {onnx_path}")


def main():
    models_dir = Path("models")
    
    # Convert baseline model
    baseline_pth = models_dir / "baseline_model.pth"
    baseline_onnx = models_dir / "baseline_model.onnx"
    
    if baseline_pth.exists():
        convert_pth_to_onnx(baseline_pth, baseline_onnx)
    else:
        print(f"  ✗ {baseline_pth} not found")
    
    # Convert improved model
    improved_pth = models_dir / "improved_model.pth"
    improved_onnx = models_dir / "improved_model.onnx"
    
    if improved_pth.exists():
        convert_pth_to_onnx(improved_pth, improved_onnx)
    else:
        print(f"  ✗ {improved_pth} not found")
    
    print("\nConversion complete!")
    print("You can now commit and push the .onnx files.")


if __name__ == "__main__":
    main()
