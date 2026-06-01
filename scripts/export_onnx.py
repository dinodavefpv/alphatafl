"""
Export AlphaTaflNet to ONNX format for native C++ ONNX Runtime inference.

Usage:
    python scripts/export_onnx.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.model.network import AlphaTaflNet


def main():
    model_path = "models/current_best.pt"
    onnx_path = "models/alphatafl.onnx"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AlphaTaflNet().to(device).eval()

    # Load weights if available
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"No weights found at {model_path}, using random init")

    # Patch forward method for ONNX export to avoid the dynamic reshape bug
    def patched_forward(self, x):
        import torch.nn.functional as F
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        
        # Policy
        p = self.policy_head(x)
        p = p.permute(0, 2, 3, 1)
        p = p.flatten(1) # [batch_size, 4840]
        
        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

    import types
    model.forward = types.MethodType(patched_forward, model)

    # Create dummy input with a batch size of 1 for tracing
    dummy_input = torch.randn(1, 14, 11, 11, device=device)

    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=15,          # Modern opset version
        do_constant_folding=True,  # Constant folding for optimization
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        dynamo=False
    )

    print(f"ONNX model saved successfully to {onnx_path}")

    # Verify model with onnx library if installed
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX checker passed")
    except ImportError:
        print("[WARNING] python-onnx package not installed; skipping checkers. (Safe to ignore)")
    except Exception as e:
        print(f"[FAIL] ONNX verification failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
