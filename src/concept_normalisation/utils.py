import torch


def get_device() -> torch.device:
    """Same device selection copy-pasted in embed_sapbert.py and
    semantic_analysis_pipeline.py: use Apple Silicon MPS if available, else CPU."""
    return torch.device("mps" if torch.mps.is_available() else "cpu")
