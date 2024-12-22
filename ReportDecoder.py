import torch
import torch.nn as nn
import torch.nn.functional as F

class ReportDecoder(nn.Module):
    """
    ReportDecoder class for generating cardiologist-style reports from shared embeddings using a Transformer-based architecture.

    Attributes:
        representation_embedding_dim (int): Dimension of the ECG representation embedding space (default: 512).
        max_length (int): Maximum length of the generated reports.
        decoder_name (str): Hugging face path of the pre-trained decoder
        decoder (nn.Module): Transformer-based decoder module.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the pre-trained model.
        vocab_size (int): vocab size of the pre-trained decoder
        decoder_input_dimension (int): Input dimension size of the pre-trained decoder
    """

    def __init__(self, decoder_name: str = 'biogpt', max_length: int = 256):
        """
        Initialize the ReportDecoder.

        Args:
            representation_embedding_dim (int, optional): Dimension of the ECG representation embedding space (default: 512).
            decoder_name (str): Hugging face path of the pre-trained decoder
            max_length (int, optional): Maximum length of the generated reports (default: 128).
        """
        super(ReportDecoder, self).__init__()
        self.max_length = max_length
        self.decoder_name = decoder_name
        
        # Load a pre-trained language model as a decoder (using GPT-2 here as a placeholder)
        self.initialize_model()

    def initialize_model(self):
        print("Initializing ReportDecoder...")
        if self.decoder_name == 'gpt2':
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
            self.vocab_size = self.decoder.config.vocab_size
            self.decoder_input_dimension = self.decoder.config.n_embd
        elif self.decoder_name == 'biogpt':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            self.decoder = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
            self.decoder.init_weights()
            self.vocab_size = self.decoder.config.vocab_size
            self.decoder_input_dimension = self.decoder.config.hidden_size
        elif self.decoder_name == 'biobart':
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            self.decoder = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
            self.vocab_size = self.decoder.config.vocab_size
            self.decoder_input_dimension = self.decoder.config.d_model
        elif self.decoder_name == 't5':
            raise ValueError(f"T5 model no longer supported for ReportDecoder")
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.vocab_size = self.decoder.config.vocab_size
            self.decoder_input_dimension = self.decoder.config.d_model
        else:
            raise ValueError(f"Unsupported model architecture: {self.decoder_name}")

        print(f"> ReportDecoder vocab_size: {self.vocab_size}")
        print(f"> ReportDecoder decoder_input_dimension: {self.decoder_input_dimension}")
        print(f"> self.tokenizer.pad_token_id: {self.tokenizer.pad_token_id}")

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, label_smoothing=0.1)


    def forward(self, representation_embedding: torch.Tensor, target_text: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ReportDecoder to generate text logits from shared metric embeddings. Only used during training.

        Args:
            representation_embedding (torch.Tensor): Shared representation embeddings from the ECGEncoder (batch_size, ECGEncoer.representation_embedding_dim). (representation_embedding_dim==self.decoder_input_dimension)
            target_text (dict[str, torch.Tensor]): Tokenized target text for teacher forcing during training.

        Returns:
            torch.Tensor: Logits for the vocabulary distribution of the generated report.
        """
        #print("ReportDecoder forward...")
        #print(f">Inputs: representation_embedding: {representation_embedding.shape}")
        if representation_embedding.size(-1) != self.decoder_input_dimension:
            raise ValueError(f"ReportDecoder expects input dimension: {representation_embedding.size(-1)} to be compatible with the pre-trained decoder input dimension: {self.decoder_input_dimension}")
        decoder_input = representation_embedding.unsqueeze(1) # (batch_size, 1, decoder_input_dimension)
        #print(f"> decoder_input: {decoder_input.shape}")
        if self.decoder_name == 't5':
            raise ValueError(f"T5 model no longer supported for ReportDecoder")
            # For T5, we need to provide both encoder and decoder inputs
            encoder_outputs = self.decoder.encoder(inputs_embeds=decoder_input)
            if target_text is not None:
                target_ids = target_text['input_ids']
                outputs = self.decoder(encoder_outputs=encoder_outputs,decoder_input_ids=target_ids,labels=target_ids)
            else:
                outputs = None
        else:
            # Expand the decoder input to match the target sequence length if target text is provided
            if target_text is not None:
                seq_length = target_text['input_ids'].size(1)
                #TODO:This is very ineffective:
                decoder_input = decoder_input.expand(-1, seq_length, -1)  # Shape: (batch_size, seq_length, decoder_input_dimension)
                #print(f"> decoder_input: {decoder_input.shape}")
            outputs = self.decoder(inputs_embeds=decoder_input, labels=target_text['input_ids'])
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs # Shape: (batch_size, seq_length, vocab_size)
        #print(f">Outputs logits: {logits.shape}")
        return logits

    def generate_report(self, representation_embedding: torch.Tensor, max_length: int = None) -> str:
        """
        Generate a report from the shared embedding using greedy decoding or beam search.

        Args:
            representation_embedding (torch.Tensor): Representation embeddings from the SharedEmbeddingSpace module of shape (batch_size, shared_embedding_dim).
            max_length (int, optional): Maximum length of the generated report (default: None, uses self.max_length).

        Returns:
            str: Generated report text.
        """
        if max_length is None:
            max_length = self.max_length

        # Expand the input to have the necessary sequence length for decoding (shape: (batch_size, 1, n_embd))
        if self.decoder_name == 't5':
            raise ValueError(f"T5 model no longer supported for ReportDecoder")
            # For T5, we need to provide encoder inputs and use the generate method
            encoder_outputs = self.decoder.encoder(inputs_embeds=representation_embedding)
            generated_ids = self.decoder.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            generated_ids = self.decoder.generate(
                inputs_embeds=representation_embedding,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        #print(f"generated_ids: {generated_ids.shape}")
        # Decode the generated token IDs into text
        report_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return report_text

    def compute_captioning_loss(self, logits: torch.Tensor, target_text: torch.Tensor) -> torch.Tensor:
        """
        Compute the captioning loss for the generated text.

        Args:
            logits (torch.Tensor): Logits from the decoder of shape (batch_size, seq_length, vocab_size).
            target_text (dict[str, torch.Tensor]): Tokenized target text for teacher forcing during training.

        Returns:
            torch.Tensor: Computed captioning loss value.
        """
        # Auto-regressive shifting the logits and target text for computing the cross-entropy loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_target_text = target_text['input_ids'][:, 1:].contiguous()
        # Flatten the logits and target text for loss computation
        captioning_loss = self.loss_fn(shift_logits.view(-1, self.vocab_size), shift_target_text.view(-1))

        return captioning_loss

    def freeze_decoder(self):
        """
        Freeze the parameters of the text encoder to prevent them from being updated during training.
        """
        print("Freezing the report decoder parameters...")
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        """
        Unfreeze the parameters of the text encoder to allow them to be updated during training.
        """
        print("Unfreezing the report decoder parameters...")
        for param in self.decoder.parameters():
            param.requires_grad = True

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

    def curr_device(self):
        return next(self.decoder.parameters()).device

    def display_model_summary(self):
        """
        Display a summary of the ReportDecoder model.
        """
        print("ReportDecoder Model Summary:")
        print(self)


# Example usage of ReportDecoder with placeholder functions:
if __name__ == "__main__":
    #TODO: update this testing
    
    # Create an instance of ReportDecoder with default parameters
    report_decoder = ReportDecoder(representation_embedding_dim=512, decoder_name='biogpt', max_length=128)
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
