import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents, weightless = False):
        super().__init__()
        self.weightless = weightless
        self.to_latents = None if self.weightless else nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = x if self.weightless else self.to_latents(x)
        return F.normalize(latents, p=2, dim=-1) 

class SharedMetricSpace(nn.Module):
    """
    SharedMetricSpace class for managing the projection of ECG and text embeddings into a shared space.

    Attributes:
        ecg_embedding_dim (int): Dimension of the ECG embeddings.
        text_embedding_dim (int): Dimension of the text embeddings.
        shared_metric_embedding_dim (int): Dimension of the shared metric embedding space.
        ecg_projection_layer (nn.Module): Linear layer for projecting ECG embeddings to the shared metric space.
        text_projection_layer (nn.Module): Linear layer for projecting text embeddings to the shared metric space.
    """

    def __init__(self, 
                 ecg_embedding_dim: int = 512, 
                 text_embedding_dim: int = 512, 
                 shared_metric_embedding_dim: int = 512):
        """
        Initialize the SharedMetricSpace.

        Args:
            ecg_embedding_dim (int, optional): Dimension of the ECG embeddings (default: 512).
            text_embedding_dim (int, optional): Dimension of the text embeddings (default: 512).
            shared_metric_embedding_dim (int, optional): Dimension of the shared metric embedding space (default: 512).
        """
        super(SharedMetricSpace, self).__init__()

        self.ecg_embedding_dim = ecg_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.shared_metric_embedding_dim = shared_metric_embedding_dim

        self.learn_ecg_projection_only = True if self.text_embedding_dim == self.shared_metric_embedding_dim else False

        # Linear layers for projecting ECG and text embeddings into the shared space
        self.ecg_projection_layer = EmbedToLatents(self.ecg_embedding_dim, self.shared_metric_embedding_dim)
        self.text_projection_layer = EmbedToLatents(self.text_embedding_dim, self.shared_metric_embedding_dim, self.learn_ecg_projection_only)
    

    def project_to_shared_metric_space(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Project ECG and text embeddings into the shared embedding space.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embedding of shape (batch_size, ecg_embedding_dim).
            text_embedding (torch.Tensor): Input text embedding of shape (batch_size, text_embedding_dim).

        Returns:
            tuple: Projected ECG and text embeddings of shape (batch_size, shared_embedding_dim).
        """
        # Project ECG and text embeddings into the shared space
        ecg_shared = self.ecg_projection_layer(ecg_embedding)  # Shape: (batch_size, shared_metric_embedding_dim)
        text_shared = self.text_projection_layer(text_embedding)  # Shape: (batch_size, shared_metric_embedding_dim)

        return ecg_shared, text_shared


    def compute_similarity(self, ecg_shared: torch.Tensor, text_shared: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between ECG and text embeddings in the shared space using cosine similarity.

        Args:
            ecg_shared (torch.Tensor): Projected ECG embeddings in the shared space.
            text_shared (torch.Tensor): Projected text embeddings in the shared space.

        Returns:
            torch.Tensor: Similarity score tensor of shape (batch_size,).
        """
        # Normalize the embeddings to unit vectors
        ecg_normalized = F.normalize(ecg_shared, p=2, dim=-1)
        text_normalized = F.normalize(text_shared, p=2, dim=-1)

        # Compute cosine similarity between the normalized embeddings
        similarity = torch.sum(ecg_normalized * text_normalized, dim=-1)  # Shape: (batch_size,)
        return similarity

    def forward(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass to project embeddings and return the shared space representations.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embedding tensor. Shape: (batch_size, ECGEncoder.representation_embedding_dim)
            text_embedding (torch.Tensor): Input text embedding tensor. Shape: (batch_size, TextEncoder.embedding_dim)

        Returns:
            tuple: Projected ECG and text embeddings in the shared space.
        """
        # Project embeddings to the shared space
        #print("SharedMetricSpace forward...")
        #print(f">Inputs: ecg_embedding: {ecg_embedding.shape}, text_embedding: {text_embedding.shape}")
        ecg_shared, text_shared = self.project_to_shared_metric_space(ecg_embedding, text_embedding)
        #print(f">Outputs: ecg_shared: {ecg_shared.shape}, text_shared: {text_shared.shape}")
        return ecg_shared, text_shared

    def freeze_projection_layers(self):
        """
        Freeze the parameters of the projection layers to prevent them from being updated during training.
        """
        print("Freezing the projection layer parameters...")
        for param in self.ecg_projection_layer.parameters():
            param.requires_grad = False
        for param in self.text_projection_layer.parameters():
            param.requires_grad = False

    def unfreeze_projection_layers(self):
        """
        Unfreeze the parameters of the projection layers to allow them to be updated during training.
        """
        print("Unfreezing the projection layer parameters...")
        for param in self.ecg_projection_layer.parameters():
            param.requires_grad = True
        for param in self.text_projection_layer.parameters():
            param.requires_grad = True

    def display_model_summary(self):
        """
        Display a summary of the SharedMetricSpace model.
        """
        print("SharedMetricSpace Model Summary:")
        print(self)

# Example usage of SharedMetricSpace with placeholder functions:
if __name__ == "__main__":
    #TODO: update this testing
    
    # Create an instance of SharedMetricSpace with default parameters
    shared_embedding_space = SharedMetricSpace(ecg_embedding_dim=512, text_embedding_dim=512, shared_embedding_dim=512)

    # Display model summary
    shared_embedding_space.display_model_summary()

    # Example input ECG and text embeddings (batch_size=8, embedding_dim=512)
    ecg_example = torch.randn(8, 512)
    text_example = torch.randn(8, 512)

    # Project embeddings and retrieve shared space representations
    ecg_shared, text_shared = shared_embedding_space(ecg_example, text_example)
    print(f"ECG Shared Embedding shape: {ecg_shared.shape}")
    print(f"Text Shared Embedding shape: {text_shared.shape}")
