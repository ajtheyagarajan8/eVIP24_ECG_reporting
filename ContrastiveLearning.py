import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearning(nn.Module):
    """
    ContrastiveLearning class for implementing contrastive learning between ECG and text embeddings.

    Attributes:
        temperature (float): Temperature parameter used to scale the logits for contrastive loss.
        similarity_metric (str): Type of similarity metric to use ('cosine' or 'dot').
    """

    def __init__(self, temperature: float = 0.07, similarity_metric: str = 'cosine'):
        """
        Initialize the ContrastiveLearning module.

        Args:
            temperature (float, optional): Temperature parameter for scaling the logits (default: 0.07).
            similarity_metric (str, optional): Similarity metric to use for computing similarity scores ('cosine' or 'dot', default: 'cosine').
        """
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.similarity_metric = similarity_metric

    def compute_similarity(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between ECG and text embeddings using the specified similarity metric.

        Args:
            ecg_embedding (torch.Tensor): Projected ECG embeddings in the shared space.
            text_embedding (torch.Tensor): Projected text embeddings in the shared space.

        Returns:
            torch.Tensor: Similarity scores of shape (batch_size, batch_size).
        """
        if self.similarity_metric == 'cosine':
            ecg_normalized = F.normalize(ecg_embedding, p=2, dim=-1)
            text_normalized = F.normalize(text_embedding, p=2, dim=-1)
            similarity_scores = torch.matmul(ecg_normalized, text_normalized.T)  # Shape: (batch_size, batch_size)
        elif self.similarity_metric == 'dot':
            similarity_scores = torch.matmul(ecg_embedding, text_embedding.T)  # Shape: (batch_size, batch_size)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        return similarity_scores

    def contrastive_loss(self, similarity_scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Compute the contrastive loss using the similarity scores.

        Args:
            similarity_scores (torch.Tensor): Similarity scores between ECG and text embeddings of shape (batch_size, batch_size).
            batch_size (int): Size of the batch (number of positive and negative pairs).

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Scale similarity scores by temperature
        scaled_similarity = similarity_scores / self.temperature  # Shape: (batch_size, batch_size)

        # Generate labels for contrastive learning (diagonal is 1, rest is 0)
        labels = torch.arange(batch_size).long().to(scaled_similarity.device)

        # Compute contrastive loss (InfoNCE loss)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scaled_similarity, labels)

        return loss

    def forward(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the contrastive loss between ECG and text embeddings.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embeddings of shape (batch_size, shared_embedding_dim).
            text_embedding (torch.Tensor): Input text embeddings of shape (batch_size, shared_embedding_dim).

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Compute similarity between ECG and text embeddings
        similarity_scores = self.compute_similarity(ecg_embedding, text_embedding)

        # Compute contrastive loss using similarity scores
        batch_size = ecg_embedding.size(0)
        loss = self.contrastive_loss(similarity_scores, batch_size)

        return loss

    def train_step(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step for contrastive learning.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embeddings.
            text_embedding (torch.Tensor): Input text embeddings.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

        Returns:
            float: Computed contrastive loss value for this step.
        """
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute the contrastive loss
        loss = self.forward(ecg_embedding, text_embedding)

        # Backpropagate the loss
        loss.backward()

        # Update the parameters using the optimizer
        optimizer.step()

        return loss.item()

    def evaluate(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """
        Evaluate the model's performance using contrastive loss.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embeddings.
            text_embedding (torch.Tensor): Input text embeddings.

        Returns:
            float: Computed contrastive loss value for evaluation.
        """
        with torch.no_grad():
            # Compute the contrastive loss without backpropagation
            loss = self.forward(ecg_embedding, text_embedding)
        return loss.item()

    def display_model_summary(self):
        """
        Display a summary of the ContrastiveLearning model.
        """
        print("ContrastiveLearning Model Summary:")
        print(self)


# Example usage of ContrastiveLearning with placeholder functions:
if __name__ == "__main__":
    # Create an instance of ContrastiveLearning with default parameters
    contrastive_learning = ContrastiveLearning(temperature=0.07, similarity_metric='cosine')

    # Display model summary
    contrastive_learning.display_model_summary()

    # Example input ECG and text embeddings (batch_size=8, embedding_dim=512)
    ecg_example = torch.randn(8, 512)
    text_example = torch.randn(8, 512)

    # Example optimizer
    optimizer = torch.optim.Adam(contrastive_learning.parameters(), lr=0.001)

    # Perform a single training step
    loss_value = contrastive_learning.train_step(ecg_embedding=ecg_example, text_embedding=text_example, optimizer=optimizer)
    print(f"Training Loss: {loss_value}")

    # Evaluate model performance
    eval_loss_value = contrastive_learning.evaluate(ecg_embedding=ecg_example, text_embedding=text_example)
    print(f"Evaluation Loss: {eval_loss_value}")
