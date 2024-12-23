import torch
import torch.nn as nn
import torch.nn.functional as F
from ECGModels.ConvTransformer1D import CTN
from torchvision import models
from pathlib import Path
class ECGEncoder(nn.Module):
    """
    ECGEncoder class for encoding ECG waveforms into a shared embedding space.

    Attributes:
        model_architecture (str): Specifies the base model architecture to use for encoding.
        embedding_dim (int): Dimension of the shared embedding space.
        pretrained (bool): Indicates if the base model should be pretrained.
        encoder (nn.Module): Placeholder for the actual model used for encoding.
    """

    def __init__(self, 
                 model_architecture: str = "ResNet101", 
                 representation_embedding_dim: int = 512,  
                 pretrained_model: str = ""):
        """
        Initialize the ECGEncoder.

        Args:
            model_architecture (str, optional): Choice of base model architecture (default: "ConvNext1D").
            representation_embedding_dim (int, optional): Dimension of the representation embedding space which links to the text decoder (default: 512).
            signal_or_image (str, optional): Specifies the type of input ('signal' or 'image', default: "signal").
            pretrained (bool, optional): Indicates if the base model should be pretrained (default: True).
        """
        super(ECGEncoder, self).__init__()
        
        self.model_architecture = model_architecture
        self.representation_embedding_dim = representation_embedding_dim
        self.pretrained_model = pretrained_model
        
        self.encoder = None

        # Initialize the encoder model based on the given architecture
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the encoder model based on the chosen architecture.
        """
        print(f"Initializing the ECGEncoder {self.model_architecture} model...")
        
        if self.model_architecture == "ConvTransformer1D":
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")
            self.encoder = CTN(d_model=256, dropout_rate=0.2,deepfeat_sz=64, classes=None)
        elif self.model_architecture == "ConvNext1D":
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=12, out_channels=64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(output_size=1),
                nn.Flatten(),
                nn.Linear(128, self.representation_embedding_dim)
            )
        elif self.model_architecture == "ResNet101":
            # Load pre-trained ResNet-101 model
            ResNet101 = models.resnet101(pretrained=True)
            
            # We don't change the fc layer anymore, but uncomment if you want to load model
            ResNet101.fc = nn.Linear(ResNet101.fc.in_features, self.representation_embedding_dim)
            #ResNet101.fc = nn.Linear(ResNet101.fc.in_features, ResNet101.fc.in_features // 2)
            
            # Load the pre-trained weights, skipping the last layer
            if self.pretrained_model != "":
                weight_file = Path(self.pretrained_model)
                if weight_file.is_file():
                    model_dict = ResNet101.state_dict()
                    pretrained_dict = torch.load(self.pretrained_model)
                    # Filter out the final fc layer weights
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
                    # Update the model's state dict with the filtered weights
                    model_dict.update(pretrained_dict)
                    ResNet101.load_state_dict(model_dict)
                else:
                    print(f"Pre-trained weights file '{weight_file}' not found. Training from scratch.")

            self.encoder = ResNet101
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ECG encoder to obtain the shared embedding.

        Args:
            x (torch.Tensor): Input tensor representing ECG waveforms. Shape: (batch_size, 3, image_width, image_height) for ECG image representation

        Returns:
            torch.Tensor: Encoded ECG embedding of shape (batch_size, representation_embedding_dim).
        """
        if self.encoder is None:
            raise ValueError("Encoder model has not been initialized.")
        #print("ECGEncoder")
        #print(f">Input x.shape: {x.shape}")
        # Forward pass through the encoder model
        x = self.encoder(x)
        #print(f">Output x.shape: {x.shape}")
        return x

    def save_model(self, file_path: str):
        """
        Save the encoder model to disk.

        Args:
            file_path (str): Path to save the model file.
        """
        print(f"Saving model to {file_path}...")
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str):
        """
        Load the encoder model from disk.

        Args:
            file_path (str): Path to the model file to load.
        """
        print(f"Loading model from {file_path}...")
        self.load_state_dict(torch.load(file_path))

    def freeze_encoder(self, freeze_metric_embedding: bool = False):
        """
        Freeze the parameters of the encoder to prevent them from being updated during training.
        """
        print("Freezing the encoder parameters...")
        for name, param in self.encoder.named_parameters():
            if not freeze_metric_embedding and 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the parameters of the encoder to allow them to be updated during training.
        """
        print("Unfreezing the encoder parameters...")
        for param in self.encoder.parameters():
            param.requires_grad = True

    def display_model_summary(self):
        """
        Display a summary of the encoder model.
        """
        print("Model Summary:")
        print(self.encoder)

# Example usage of ECGEncoder with placeholder functions:
if __name__ == "__main__":
    # Create an instance of ECGEncoder with placeholder parameters
    ecg_encoder = ECGEncoder(model_architecture="ResNet101", representation_embedding_dim=512, pretrained_model="resnetPTBXL_weights.pth")

    # Print the model summary
    ecg_encoder.display_model_summary()

    # Example input tensor for the forward pass (batch_size=8, channels=12, sequence_length=1000)
    example_input = torch.randn(8, 12, 1000)
    
    # Perform a forward pass
    output = ecg_encoder(example_input)
    print(f"Output shape: {output.shape}")
