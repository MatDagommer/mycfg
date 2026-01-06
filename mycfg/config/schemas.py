from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration schema for training parameters."""
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    data_dir: str = "./data"
    save_model: bool = True
    model_path: str = "checkpoints/fashion_mnist_cnn.pth"
    download: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")