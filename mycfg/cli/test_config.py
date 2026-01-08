"""Test configuration loading and merging for training a CNN on Fashion-MNIST."""

from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger

from ..config.schemas import TrainingConfig
from ..config.utils import (
    auto_config_cli,
    get_current_config_overrides,
    load_config_with_overrides,
)

app = typer.Typer(help="Train a CNN model on the Fashion-MNIST dataset.")

@app.command()
@auto_config_cli(TrainingConfig)
def test_config(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", \
        help="Path to configuration YAML file"),
    save_config: Optional[str] = typer.Option(None, \
        help="Path to save the final configuration")
):
    """Train a CNN model on Fashion-MNIST dataset."""

    # Determine config file path
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = None

    # Use overrides from the decorator
    overrides = get_current_config_overrides()

    # Load and merge configuration
    save_config_path = Path(save_config) if save_config else None
    config = load_config_with_overrides(
        TrainingConfig,
        config_file,
        overrides,
        save_config_path
    )

    print("Final Merged Configuration:")
    print(config)

    logger.info(f"Configuration loaded: {config}")

    device = torch.device("mps" if torch.backends.mps.is_available() \
        else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

if __name__ == "__main__":
    typer.run(test_config)
