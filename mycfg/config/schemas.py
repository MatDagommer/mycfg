from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """Configuration schema for training parameters."""
    epochs: int = field(default=10, metadata={"help": "Number of training epochs to run"})
    batch_size: int = field(default=64, metadata={"help": "Batch size for training data loader"})
    learning_rate: float = field(default=0.001, metadata={"help": "Learning rate for the optimizer"})
    data_dir: str = field(default="./data", metadata={"help": "Directory to store/load dataset"})
    save_model: bool = field(default=True, metadata={"help": "Whether to save the trained model"})
    model_path: str = field(default="checkpoints/fashion_mnist_cnn.pth", metadata={"help": "Path to save/load the trained model"})
    download: bool = field(default=False, metadata={"help": "Whether to download the dataset if not present"})
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")