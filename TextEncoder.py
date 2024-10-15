import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """
    TextEncoder class for encoding cardiologist reports into a shared embedding space using a pre-trained text model.

    Attributes:
        pretrained_model (str): Name or path of the pre-trained text model to use.
        embedding_dim (int): Dimension of the shared embedding space.
        model (nn.Module): Loaded pre-trained text model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.
    """
    
    def __init__(self, 
                 pretrained_model: str = "michiyasunaga/BioLinkBERT-base", 
                 embedding_dim: int = 512):
        """
        Initialize the TextEncoder.

        Args:
            pretrained_model (str, optional): Name or path of the pre-trained text model (default: "michiyasunaga/BioLinkBERT-base").
            embedding_dim (int, optional): Dimension of the shared embedding space (default: 512).
        """
        super(TextEncoder, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.embedding_dim = embedding_dim
        
        # Placeholder for model and tokenizer
        self.model = None
        self.tokenizer = None

        # Initialize the model and tokenizer
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the pre-trained text model and its tokenizer.
        """
        print(f"Loading pre-trained model: {self.pretrained_model}...")
        # Load the pre-trained text model and corresponding tokenizer
        self.model = AutoModel.from_pretrained(self.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        # Optional: Project output to the specified embedding dimension if needed
        self.embedding_projection = nn.Linear(self.model.config.hidden_size, self.embedding_dim)

    def forward(self, text_input: str) -> torch.Tensor:
        """
        Forward pass through the TextEncoder to obtain text embeddings.

        Args:
            text_input (str): Input text string to encode.

        Returns:
            torch.Tensor: Encoded text embedding of shape (batch_size, embedding_dim).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer has not been initialized.")

        # Tokenize the input text and create a tensor representation
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        inputs.to(next(self.parameters()).device)
        
        #print(f"TextEncoder inputs after tokenization: {inputs['input_ids'].shape}")
        # Forward pass through the model to get hidden states
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the last hidden state (or use [CLS] token embedding)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        # Optional: Project to the shared embedding dimension
        text_embedding = self.embedding_projection(last_hidden_state[:, 0, :])  # Shape: (batch_size, embedding_dim)


        return text_embedding

    def save_model(self, file_path: str):
        """
        Save the text encoder model to disk.

        Args:
            file_path (str): Path to save the model file.
        """
        print(f"Saving model to {file_path}...")
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str):
        """
        Load the text encoder model from disk.

        Args:
            file_path (str): Path to the model file to load.
        """
        print(f"Loading model from {file_path}...")
        self.load_state_dict(torch.load(file_path))

    def freeze_encoder(self):
        """
        Freeze the parameters of the text encoder to prevent them from being updated during training.
        """
        print("Freezing the text encoder parameters...")
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the parameters of the text encoder to allow them to be updated during training.
        """
        print("Unfreezing the text encoder parameters...")
        for param in self.model.parameters():
            param.requires_grad = True

    def display_model_summary(self):
        """
        Display a summary of the text encoder model.
        """
        print("Model Summary:")
        print(self.model)

# Example usage of TextEncoder with placeholder functions:
if __name__ == "__main__":
    # Create an instance of TextEncoder with default parameters
    text_encoder = TextEncoder(pretrained_model="michiyasunaga/BioLinkBERT-base", embedding_dim=512)

    # Print the model summary
    text_encoder.display_model_summary()

    # Example input text
    example_text = "Patient exhibits signs of atrial fibrillation with irregular R-R intervals."
    
    # Perform a forward pass to encode the text
    text_embedding = text_encoder(example_text)
    print(f"Text embedding shape: {text_embedding.shape}")
