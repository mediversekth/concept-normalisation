import torch


def get_device() -> torch.device:
    """Same device selection copy-pasted in embed_sapbert.py and
    semantic_analysis_pipeline.py: 
        use Apple Silicon MPS if available, 
        use CUDA if available,
        else CPU."""
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
