import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer

#facebook/bart-base
class TextEncoder(nn.Module):
    """
    TextEncoder class for encoding cardiologist reports into a representation embedding using a pre-trained model and projection layers.

    Attributes:
        pretrained_model (str): Name or path of the pre-trained text model to use. Supports: "michiyasunaga/BioLinkBERT-base", "t5-small", "facebook/bart-base"
        embedding_dim (int): Dimension of the pre-trained text encoder embedding (output of TextEncoder).
        model (nn.Module): Loaded pre-trained text encoder model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.
        hidden_size (int): Size of the output layer of the pre-trained encoder
        representation_embedding_projection_1 (nn.Module): first text representation embedding layer
        representation_embedding_projection_2 (nn.Module): second and final text representation embedding layer
    """
    
    def __init__(self, pretrained_model: str = "michiyasunaga/BioLinkBERT-base"):
        """
        Args:
            pretrained_model (str, optional): Name or path of the pre-trained text model (default: "michiyasunaga/BioLinkBERT-base").
        """
        super(TextEncoder, self).__init__()
        
        self.pretrained_model = pretrained_model
        
        # Placeholder for model and tokenizer
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None

        # Initialize the model and tokenizer
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the pre-trained text model and its tokenizer.
        """
        print("Initializing TextEncoder...")
        if "t5" in self.pretrained_model:
            print(f"Loading T5 encoder model: {self.pretrained_model}...")
            self.model = T5EncoderModel.from_pretrained(self.pretrained_model)
            self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model)
            #TODO: set something like model_max_length?
        else:
            print(f"Loading pre-trained model: {self.pretrained_model}...")
            self.model = AutoModel.from_pretrained(self.pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
            #self.tokenizer.model_max_length = 1000
        
        if "bart" in self.pretrained_model:
            self.model = self.model.encoder
        self.embedding_dim = self.model.config.hidden_size
        print(f"> TextEncoder embedding_dim: {self.embedding_dim}")

    def forward(self, text_input: str) -> torch.Tensor:
        """
        Forward pass through the TextEncoder to obtain text embeddings.

        Args:
            text_input List[str]: List of input text strings to encode. len(text_input)==batch_size

        Returns:
            torch.Tensor: Encoded text embedding of shape (batch_size, embedding_dim).
        """
        #print("TextEncoder.forward...")
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer has not been initialized.")
        
        # Tokenize the input text and create a tensor representation
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        #print(f"> TextEncoder tokenized len(inputs['input_ids']): {len(inputs['input_ids'])}")
        #print(f"> inputs['input_ids'][0].shape: {inputs['input_ids'][0].shape}")
        
        inputs.to(next(self.parameters()).device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        #print(f"> TextEncoder outputs.last_hidden_state.shape: {outputs.last_hidden_state.shape}")
        
        # T5 outputs a different structure than standard BERT-like models
        if "t5" in self.pretrained_model:
            # T5 returns only hidden states in the outputs
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            attention_mask = inputs['attention_mask']  # Shape: (batch_size, sequence_length)

            # Expand attention mask to match the dimensions of last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # Shape: (batch_size, sequence_length, hidden_size)

            # Perform mean pooling to get single embedding 
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)  # shape (batch_size, hidden_size)
            sum_mask = input_mask_expanded.sum(dim=1)  # shape: (batch_size, hidden_size)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # avoid division by zero
            mean_pooled = sum_embeddings / sum_mask  # shape: (batch_size, hidden_size)
            text_embedding = mean_pooled
            #print(f"> mean_pooled.shape: {mean_pooled.shape}")
        else:
            last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            # Use the [CLS] token representation (first token)
            cls_embedding = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            #print(f"> cls_embedding.shape: {cls_embedding.shape}")
            text_embedding = cls_embedding
        #print(f"> text_embedding.shape: {text_embedding.shape}")
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
    pretrained_model = "michiyasunaga/BioLinkBERT-base"
    #pretrained_model = "t5-small"
    #pretrained_model = "facebook/bart-base"
    text_encoder = TextEncoder(pretrained_model=pretrained_model, embedding_dim=512)

    # Print the model summary
    #text_encoder.display_model_summary()

    # Example input text
    example_text = "Patient exhibits signs of atrial fibrillation with irregular R-R intervals."
    
    # Perform a forward pass to encode the text
    text_embedding = text_encoder(example_text)
    print(f"Text embedding shape: {text_embedding.shape}")
