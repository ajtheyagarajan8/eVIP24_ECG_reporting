import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContrastiveLearning(nn.Module):
    """
    ContrastiveLearning class for implementing contrastive loss between ECG and text metric embeddings.

    Attributes:
        temperature (float): Temperature parameter used to scale the logits for contrastive loss.
        similarity_metric (str): Type of similarity metric to use ('cosine' or 'dot').
        loss_fn (nn.CrossEntropyLoss)
        temperature
    """

    def __init__(self, temperature: float = 1.0, similarity_metric: str = 'dot'):
        """
        Args:
            temperature (float, optional): Temperature parameter for scaling the logits (default: 0.07).
            similarity_metric (str, optional): Similarity metric to use for computing similarity scores ('cosine' or 'dot', default: 'cosine').
        """
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.similarity_metric = similarity_metric
        self.loss_fn = nn.CrossEntropyLoss()
        #TODO make temperature a learnable parameter:
        #self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.temperature = 1.0

    def compute_similarity(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between ECG and text embeddings using the specified similarity metric.

        Args:
            ecg_embedding (torch.Tensor): Projected ECG embeddings in the shared metric space.
            text_embedding (torch.Tensor): Projected text embeddings in the shared metric space.

        Returns:
            torch.Tensor: Similarity scores of shape (batch_size, batch_size).
        """
        if self.similarity_metric == 'dot':
            similarity_scores = torch.einsum('i d, j d -> i j', text_embedding, ecg_embedding)
            #TODO make temperature a learnable parameter:
            #similarity_scores = similarity_scores * self.temperature.exp()
            similarity_scores = similarity_scores * math.exp(self.temperature)
            
            #similarity_scores = torch.matmul(ecg_embedding, text_embedding.T)  # Shape: (batch_size, batch_size)
            '''
                Each row i in the scale_similarity matrix contains the similarity scores between the i-th ECG 
                embedding and all text embeddings (including its correct match).The diagonal element in the 
                matrix corresponds to the similarity between the matching positive pairs (i.e., ecg_embedding[i] with 
                text_embedding[i]). The off-diagonal elements represent the similarities with non-matching 
                embeddings, which act as negative samples.
            '''
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
        
        # Generate labels for contrastive learning (diagonal is 1, rest is 0)
        labels = torch.arange(batch_size, device=similarity_scores.device).long() #Shape (batch_size,)
        '''
            This creates a labels [0,1,2,3,...,batch_size-1] where each label represents the position of 
            the corresponding "true positive" match. Specifically, labels[i] is set to i, which means that 
            for the i-th ECG embedding, the correct text embedding (the positive pair) is expected to be 
            at position i in the list of text embeddings.
        '''
        #print(f"labels: {labels}")
        #print(f"scaled_similarity: {scaled_similarity}")
        loss = (self.loss_fn(similarity_scores, labels) + self.loss_fn(similarity_scores.t(), labels))*0.5
        '''
            The cross-entropy loss takes scaled_similarity as input logits and labels as the ground truth.
            Cross-entropy loss works by comparing the similarity scores across all potential pairs for a 
            given ECG embedding. It tries to maximize the score of the correct match (the diagonal element) 
            and minimize the scores of incorrect matches (off-diagonal elements). Using labels[i] = i means 
            that the model is penalized if the similarity of the positive pair is not the highest in its row.
        '''
        return loss

    def forward(self, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the contrastive loss between ECG and text embeddings.

        Args:
            ecg_embedding (torch.Tensor): Input ECG embeddings of shape (batch_size, shared_metric_embedding_dim).
            text_embedding (torch.Tensor): Input text embeddings of shape (batch_size, shared_metric_embedding_dim).

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        #print("ContrastiveLearning forward...")
        #print(f">Inputs ecg_embedding: {ecg_embedding.shape}, text_embedding: {text_embedding.shape}")
        # Compute similarity between ECG and text embeddings
        similarity_scores = self.compute_similarity(ecg_embedding, text_embedding)
        #print(f">Similarity scores: {similarity_scores.shape}")
        # Compute contrastive loss using similarity scores
        batch_size = ecg_embedding.size(0)
        loss = self.contrastive_loss(similarity_scores, batch_size)
        #print(f">Outputs loss: {loss}")
        return loss

    def display_model_summary(self):
        """
        Display a summary of the ContrastiveLearning model.
        """
        print("ContrastiveLearning Model Summary:")
        print(self)


if __name__ == "__main__":
    #TODO: update this testing

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
