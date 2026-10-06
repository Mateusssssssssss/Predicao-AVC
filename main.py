"""Orquestrador do projeto e entrada ASGI: uvicorn main:app --reload."""
from src.api import app
from src.models.train import run_training

if __name__ == "__main__":
    run_training()
