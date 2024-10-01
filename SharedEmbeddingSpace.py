import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEmbeddingSpace(nn.Module):
    """
    SharedEmbeddingSpace class for managing the projection of ECG and text embeddings into a shared space.

    Attributes:
        ecg_embedding_dim (int): Dimension of the ECG embeddings.
        text_embedding_dim (int): Dimension of the text embeddings.
        shared_embedding_dim (int): Dimension of the shared embedding space.
        ecg_projection_layer (nn.Module): Linear layer for projecting ECG embeddings to the shared space.
        text_projection_layer (nn.Module): Linear layer for projecting text embeddings to the shared space.
    """

    def __init__(self, 
                 ecg_embedding_dim: int = 512, 
                 text_embedding_dim: int = 512, 
                 shared_embedding_dim: int = 512):
        """
        Initialize the SharedEmbeddingSpace.

        Args:
            ecg_embedding_dim (int, optional): Dimension of the ECG embeddings (default: 512).
            text_embedding_dim (int, optional): Dimension of the text embeddings (default: 512).
            shared_embedding_dim (int, optional): Dimension of the shared embedding space (default: 512).
        """
        super(SharedEmbeddingSpace, self).__init__()

        self.ecg_embedding_dim = ecg_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.shared_embedding_dim = shared_embedding_dim

        # Linear layers for projecting ECG and text embeddings into the shared space
        self.ecg_projection_layer = nn.Linear(self.ecg_embedding_dim, self.shared_embedding_dim)
        self.text_projection_layer = nn.Linear(self.text_embedding_dim, self.shared_embedding_dim)

    def project_to_shared_space(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Project ECG and text embeddings into the shared embedding space.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embedding of shape (batch_size, ecg_embedding_dim).
            text_embedding (torch.Tensor): Input text embedding of shape (batch_size, text_embedding_dim).

        Returns:
            tuple: Projected ECG and text embeddings of shape (batch_size, shared_embedding_dim).
        """
        # Project ECG and text embeddings into the shared space
        ecg_shared = self.ecg_projection_layer(ecg_embedding)  # Shape: (batch_size, shared_embedding_dim)
        text_shared = self.text_projection_layer(text_embedding)  # Shape: (batch_size, shared_embedding_dim)

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
            ecg_embedding (torch.Tensor): Input ECG embedding tensor.
            text_embedding (torch.Tensor): Input text embedding tensor.

        Returns:
            tuple: Projected ECG and text embeddings in the shared space.
        """
        # Project embeddings to the shared space
        ecg_shared, text_shared = self.project_to_shared_space(ecg_embedding, text_embedding)

        # Return the projected embeddings to be used as inputs for downstream tasks
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
        Display a summary of the SharedEmbeddingSpace model.
        """
        print("SharedEmbeddingSpace Model Summary:")
        print(self)

# Example usage of SharedEmbeddingSpace with placeholder functions:
if __name__ == "__main__":
    # Create an instance of SharedEmbeddingSpace with default parameters
    shared_embedding_space = SharedEmbeddingSpace(ecg_embedding_dim=512, text_embedding_dim=512, shared_embedding_dim=512)

    # Display model summary
    shared_embedding_space.display_model_summary()

    # Example input ECG and text embeddings (batch_size=8, embedding_dim=512)
    ecg_example = torch.randn(8, 512)
    text_example = torch.randn(8, 512)

    # Project embeddings and retrieve shared space representations
    ecg_shared, text_shared = shared_embedding_space(ecg_example, text_example)
    print(f"ECG Shared Embedding shape: {ecg_shared.shape}")
    print(f"Text Shared Embedding shape: {text_shared.shape}")
