import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import typer
from loguru import logger
from pyprojroot import here
from pathlib import Path
from typing import Optional

from ..models.cnn import FashionMNISTCNN
from ..config.schemas import TrainingConfig
from ..config.utils import load_config_with_overrides, create_cli_overrides_from_schema

app = typer.Typer(help="Train a CNN model on the Fashion-MNIST dataset.")

@app.command()
def train_model(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to configuration YAML file"),
    save_config: Optional[str] = typer.Option(None, help="Path to save the final configuration")
):
    """Train a CNN model on Fashion-MNIST dataset."""
    
    # Determine config file path
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = None
    
    # Create overrides from CLI arguments using locals() and config schema
    overrides = create_cli_overrides_from_schema(TrainingConfig, **locals())
    
    # Load and merge configuration
    save_config_path = Path(save_config) if save_config else None
    config = load_config_with_overrides(
        TrainingConfig,
        config_file,
        overrides,
        save_config_path
    )
    
    logger.info(f"Configuration loaded: {config}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=config.download
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root=config.data_dir,
        train=False,
        transform=transform,
        download=config.download
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    logger.info(f"Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        logger.info(f'Epoch {epoch+1}/{config.epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    if config.save_model:
        torch.save(model.state_dict(), str(here() / config.model_path))
        logger.info(f"Model saved to {str(here() / config.model_path)}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    typer.run(train_model)