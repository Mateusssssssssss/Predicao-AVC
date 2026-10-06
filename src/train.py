"""Backward-compatible training entry point for earlier project revisions."""
from src.data.balancing import SmotencUnderSampler  # keeps older joblib bundles loadable
from src.models.train import run_training

if __name__ == "__main__":
    run_training()
