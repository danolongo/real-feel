"""
Exports rf.v1.0.0.pt to rf.v1.0.0.onnx for use with onnxruntime in the pipeline.
"""

import sys
import torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from setup.config import ModelConfig, EnsembleConfig
from setup.model import CLSMaxPoolEnsemble

PT_PATH = "rf.v1.0.0.pt"
ONNX_PATH = "rf.v1.0.0.onnx"
MAX_SEQ_LEN = 128

def load_model(pt_path: str) -> CLSMaxPoolEnsemble:
    config = ModelConfig()
    ensemble_config = EnsembleConfig()
    model = CLSMaxPoolEnsemble(config, ensemble_config)

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)

    # checkpoint may be raw state_dict or wrapped in a dict
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model

def export(model: CLSMaxPoolEnsemble, onnx_path: str):
    # dummy inputs matching model's expected shape
    dummy_input_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    import os
    os.chdir(__import__('pathlib').Path(__file__).parent)

    print("Loading model...")
    model = load_model(PT_PATH)
    print("Exporting to ONNX...")
    export(model, ONNX_PATH)
