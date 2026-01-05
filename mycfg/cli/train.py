import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import typer
from loguru import logger
from pyprojroot import here

from ..models.cnn import FashionMNISTCNN

app = typer.Typer(help="Train a CNN model on the Fashion-MNIST dataset.")

@app.command()
def train_model(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    data_dir: str = typer.Option("./data", help="Directory to store/load Fashion-MNIST data"),
    save_model: bool = typer.Option(True, help="Save trained model"),
    model_path: str = typer.Option("checkpoints/fashion_mnist_cnn.pth", help="Path to save the model"),
    download: bool = typer.Option(False, help="Download the dataset if not present")
):
    """Train a CNN model on Fashion-MNIST dataset."""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
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
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
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
        
        logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    if save_model:
        torch.save(model.state_dict(), str(here() / model_path))
        logger.info(f"Model saved to {str(here() / model_path)}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    typer.run(train_model)