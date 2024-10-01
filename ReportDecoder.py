import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ReportDecoder(nn.Module):
    """
    ReportDecoder class for generating cardiologist-style reports from shared embeddings using a Transformer-based architecture.

    Attributes:
        shared_embedding_dim (int): Dimension of the shared embedding space.
        vocab_size (int): Size of the vocabulary for the decoder.
        max_length (int): Maximum length of the generated reports.
        num_layers (int): Number of Transformer layers.
        decoder (nn.Module): Transformer-based decoder module.
        embedding_projection (nn.Linear): Linear layer to project shared embeddings to decoder input dimension.
    """

    def __init__(self, 
                 shared_embedding_dim: int = 512, 
                 vocab_size: int = 30522, 
                 max_length: int = 128, 
                 num_layers: int = 6):
        """
        Initialize the ReportDecoder.

        Args:
            shared_embedding_dim (int, optional): Dimension of the shared embedding space (default: 512).
            vocab_size (int, optional): Size of the vocabulary for the decoder (default: 30522).
            max_length (int, optional): Maximum length of the generated reports (default: 128).
            num_layers (int, optional): Number of Transformer layers in the decoder (default: 6).
        """
        super(ReportDecoder, self).__init__()

        self.shared_embedding_dim = shared_embedding_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers

        # Load a pre-trained language model as a decoder (using GPT-2 here as a placeholder)
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Linear layer to project shared embeddings to the decoder input dimension
        self.embedding_projection = nn.Linear(shared_embedding_dim, self.decoder.config.n_embd)

    def forward(self, shared_embedding: torch.Tensor, target_text: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the ReportDecoder to generate text from shared embeddings.

        Args:
            shared_embedding (torch.Tensor): Shared embeddings from the SharedEmbeddingSpace module of shape (batch_size, shared_embedding_dim).
            target_text (torch.Tensor, optional): Tokenized target text for teacher forcing during training (default: None).

        Returns:
            torch.Tensor: Logits for the vocabulary distribution of the generated report.
        """
        # Project the shared embedding to the decoder's input embedding dimension
        decoder_input = self.embedding_projection(shared_embedding)  # Shape: (batch_size, n_embd)

        # Expand the decoder input to match the target sequence length if target text is provided
        if target_text is not None:
            batch_size = shared_embedding.size(0)
            seq_length = target_text.size(1)
            decoder_input = decoder_input.unsqueeze(1).expand(-1, seq_length, -1)  # Shape: (batch_size, seq_length, n_embd)

        # Forward pass through the decoder
        outputs = self.decoder(inputs_embeds=decoder_input, labels=target_text)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        return logits

    def generate_report(self, shared_embedding: torch.Tensor, max_length: int = None) -> str:
        """
        Generate a report from the shared embedding using greedy decoding or beam search.

        Args:
            shared_embedding (torch.Tensor): Shared embeddings from the SharedEmbeddingSpace module of shape (batch_size, shared_embedding_dim).
            max_length (int, optional): Maximum length of the generated report (default: None, uses self.max_length).

        Returns:
            str: Generated report text.
        """
        if max_length is None:
            max_length = self.max_length

        # Project the shared embedding to the decoder's input embedding dimension
        decoder_input = self.embedding_projection(shared_embedding)  # Shape: (batch_size, n_embd)

        # Expand the input to have the necessary sequence length for decoding (shape: (batch_size, 1, n_embd))
        decoder_input = decoder_input.unsqueeze(1)

        # Generate tokens using the decoder
        generated_ids = self.decoder.generate(
            inputs_embeds=decoder_input,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode the generated token IDs into text
        report_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return report_text

    def compute_captioning_loss(self, logits: torch.Tensor, target_text: torch.Tensor) -> torch.Tensor:
        """
        Compute the captioning loss for the generated text.

        Args:
            logits (torch.Tensor): Logits from the decoder of shape (batch_size, seq_length, vocab_size).
            target_text (torch.Tensor): Tokenized target text of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Computed captioning loss value.
        """
        # Shift the logits and target text for computing the cross-entropy loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_target_text = target_text[:, 1:].contiguous()

        # Flatten the logits and target text for loss computation
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        captioning_loss = loss_fn(shift_logits.view(-1, self.vocab_size), shift_target_text.view(-1))

        return captioning_loss

    def save_model(self, file_path: str):
        """
        Save the report decoder model to disk.

        Args:
            file_path (str): Path to save the model file.
        """
        print(f"Saving report decoder model to {file_path}...")
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str):
        """
        Load the report decoder model from disk.

        Args:
            file_path (str): Path to the model file to load.
        """
        print(f"Loading report decoder model from {file_path}...")
        self.load_state_dict(torch.load(file_path))

    def display_model_summary(self):
        """
        Display a summary of the ReportDecoder model.
        """
        print("ReportDecoder Model Summary:")
        print(self)


# Example usage of ReportDecoder with placeholder functions:
if __name__ == "__main__":
    # Create an instance of ReportDecoder with default parameters
    report_decoder = ReportDecoder(shared_embedding_dim=512, vocab_size=30522, max_length=128, num_layers=6)

    # Display model summary
    report_decoder.display_model_summary()

    # Example input shared embedding (batch_size=8, shared_embedding_dim=512)
    example_shared_embedding = torch.randn(8, 512)

    # Generate a report from the shared embedding
    generated_report = report_decoder.generate_report(example_shared_embedding)
    print(f"Generated Report: {generated_report}")

    # Example target text for computing captioning loss (batch_size=8, seq_length=128)
    example_target_text = torch.randint(0, 30522, (8, 128))

    # Forward pass and compute logits
    logits = report_decoder(example_shared_embedding, target_text=example_target_text)
    print(f"Logits shape: {logits.shape}")

    # Compute captioning loss
    captioning_loss_value = report_decoder.compute_captioning_loss(logits, example_target_text)
    print(f"Captioning Loss: {captioning_loss_value.item()}")
