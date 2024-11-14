import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

class TrainingFramework:
    """
    TrainingFramework class for managing the end-to-end training and validation process.

    Attributes:
        ecg_encoder (nn.Module): Instance of the ECGEncoder module.
        text_encoder (nn.Module): Instance of the TextEncoder module.
        shared_embedding_space (nn.Module): Instance of the SharedEmbeddingSpace module.
        contrastive_learning (nn.Module): Instance of the ContrastiveLearning module.
        report_decoder (nn.Module): Instance of the ReportDecoder module.
        device (torch.device): Device to run the training on (CPU or GPU).
        training_params (dict): Dictionary containing training parameters (e.g., learning rate, batch size).
    """

    def __init__(self, 
                 ecg_encoder: nn.Module, 
                 text_encoder: nn.Module, 
                 shared_embedding_space: nn.Module, 
                 contrastive_learning: nn.Module, 
                 report_decoder: nn.Module,
                 training_params: dict = None):
        """
        Initialize the TrainingFramework.

        Args:
            ecg_encoder (nn.Module): Initialized ECGEncoder module.
            text_encoder (nn.Module): Initialized TextEncoder module.
            shared_embedding_space (nn.Module): Initialized SharedEmbeddingSpace module.
            contrastive_learning (nn.Module): Initialized ContrastiveLearning module.
            report_decoder (nn.Module): Initialized ReportDecoder module.
            training_params (dict, optional): Dictionary containing training parameters (default: None).
        """
        self.epoch = 0
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder
        self.shared_embedding_space = shared_embedding_space
        self.contrastive_learning = contrastive_learning
        self.report_decoder = report_decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache() 
        self.ecg_encoder.to(self.device)
        self.text_encoder.to(self.device)
        self.shared_embedding_space.to(self.device)
        self.report_decoder.to(self.device)
        # Set default training parameters if not provided
        if training_params is None:
            self.training_params = {
                'learning_rate': 1e-4,
                'num_epochs': 2,
                'weight_decay': 1e-5
            }
        else:
            self.training_params = training_params

        # Optimizers for each component
        self.optimizer = self.initialize_optimizers()

    def initialize_optimizers(self):
        """
        Initialize optimizers for each module in the framework.

        Returns:
            dict: Dictionary of optimizers for each module.
        """
        ecg_optimizer = optim.Adam(self.ecg_encoder.parameters(), lr=self.training_params['learning_rate'], weight_decay=self.training_params['weight_decay'])
        text_optimizer = optim.Adam(self.text_encoder.parameters(), lr=self.training_params['learning_rate'], weight_decay=self.training_params['weight_decay'])
        shared_space_optimizer = optim.Adam(self.shared_embedding_space.parameters(), lr=self.training_params['learning_rate'], weight_decay=self.training_params['weight_decay'])
        report_decoder_optimizer = optim.Adam(self.report_decoder.parameters(), lr=self.training_params['learning_rate'], weight_decay=self.training_params['weight_decay'])

        return {
            'ecg_optimizer': ecg_optimizer,
            'text_optimizer': text_optimizer,
            'shared_space_optimizer': shared_space_optimizer,
            'report_decoder_optimizer': report_decoder_optimizer
        }

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        self.ecg_encoder.train()
        self.text_encoder.train()
        self.shared_embedding_space.train()
        self.contrastive_learning.train()
        self.report_decoder.train()

        total_contrastive_loss = 0.0
        total_ecg_captioning_loss = 0.0
        total_text_captioning_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)
        report_comparison = ""

        for batch_idx, batch_data in enumerate(dataloader):
            # Load data and move to device
            ecg_waveforms = batch_data['waveform'].to(self.device)
            text_reports = batch_data['text']#.to(self.device)

            # Encode ECG and text into embeddings
            ecg_embeddings = self.ecg_encoder(ecg_waveforms)
            text_embeddings = self.text_encoder(text_reports)
            # Get shared embeddings using the shared embedding space
            ecg_shared, text_shared = self.shared_embedding_space(ecg_embeddings, text_embeddings)
            # Compute contrastive loss
            contrastive_loss = self.contrastive_learning(ecg_shared, text_shared)
            total_contrastive_loss += contrastive_loss.detach().item()

            #convert text_reports to tokens
            text_reports_tokenized = self.report_decoder.tokenizer(text_reports, return_tensors="pt", padding=True, truncation=True)
            text_reports_tokenized.to(next(self.report_decoder.parameters()).device) #send tokens to same device as decoder
            
            # Compute report generation loss using shared embeddings
            ecg_logits = self.report_decoder(shared_embedding=ecg_shared, target_text=text_reports_tokenized)
            ecg_captioning_loss = self.report_decoder.compute_captioning_loss(ecg_logits, target_text=text_reports_tokenized['input_ids'])
            text_logits = self.report_decoder(shared_embedding=text_shared, target_text=text_reports_tokenized)
            text_captioning_loss = self.report_decoder.compute_captioning_loss(text_logits, target_text=text_reports_tokenized['input_ids'])
            total_ecg_captioning_loss += ecg_captioning_loss.detach().item()
            total_text_captioning_loss += text_captioning_loss.detach().item()
            # Combine losses
            total_loss_value = contrastive_loss + ecg_captioning_loss + text_captioning_loss

            # Backpropagate and optimize each component
            self.optimizer['ecg_optimizer'].zero_grad()
            self.optimizer['text_optimizer'].zero_grad()
            self.optimizer['shared_space_optimizer'].zero_grad()
            self.optimizer['report_decoder_optimizer'].zero_grad()

            total_loss_value.backward()
            #self.progressive_unfreeze()
            self.apply_gradient_clipping()

            self.optimizer['ecg_optimizer'].step()
            self.optimizer['text_optimizer'].step()
            self.optimizer['shared_space_optimizer'].step()
            self.optimizer['report_decoder_optimizer'].step()

            # Accumulate loss for reporting
            total_loss += total_loss_value.detach().item()
            print(f"Epoch [{epoch + 1}/{self.training_params['num_epochs']}], Batch [{batch_idx + 1}/{num_batches}], Loss: {total_loss_value.item()}")
            if batch_idx % 10 == 0:
                    text_example_report = self.report_decoder.generate_report(text_shared[0].unsqueeze(0))
                    ecg_example_report = self.report_decoder.generate_report(ecg_shared[0].unsqueeze(0))
                    
                    actual_report = f">>Actual report: \"{text_reports[0]}\"\n"
                    text_predicted_report = f">>Text predicted report: \"{text_example_report}\"\n"
                    ecg_predicted_report = f">>ECG predicted report: \"{ecg_example_report}\"\n"
                    print(actual_report)
                    print(text_predicted_report)
                    print(ecg_predicted_report)
                    report_comparison+=f">Batch: {batch_idx}\n"
                    report_comparison+=actual_report
                    report_comparison+=text_predicted_report
                    report_comparison+=ecg_predicted_report

        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_ecg_captioning_loss = total_ecg_captioning_loss / num_batches
        avg_text_captioning_loss = total_text_captioning_loss / num_batches
        avg_total_loss = total_loss / num_batches
        loss_df = pd.DataFrame({
            "epoch": epoch,
            "avg_contrastive_loss": [avg_contrastive_loss],
            "avg_ecg_captioning_loss": [avg_ecg_captioning_loss],
            "avg_text_captioning_loss": [avg_text_captioning_loss],
            "avg_total_loss": [avg_total_loss]
        })
        return loss_df, report_comparison

    def initial_llm_freeze(self):
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False

        for param in self.report_decoder.decoder.parameters():
            param.requires_grad = False

    def progressive_unfreeze(self):
        if self.epoch < 10:
            for param in self.text_encoder.model.encoder.layer[-2:].parameters():
                param.requires_grad = True  # Unfreeze top 8 layers of text encoder
            for param in self.report_decoder.decoder.biogpt.layers[-2:].parameters():
                param.requires_grad = True  # Unfreeze top 8 layers of report decoder
        if self.epoch < 50 and self.epoch >= 10:
            for param in self.text_encoder.model.encoder.layer[-4:].parameters():
                param.requires_grad = True  # Unfreeze top 8 layers of text encoder
            for param in self.report_decoder.decoder.biogpt.layers[-4:].parameters():
                param.requires_grad = True  # Unfreeze top 8 layers of report decoder
        else:
            for param in self.text_encoder.model.parameters():
                param.requires_grad = True
            for param in self.report_decoder.decoder.parameters():
                param.requires_grad = True

    def apply_gradient_clipping(self):
        torch.nn.utils.clip_grad_norm(self.report_decoder.decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm(self.text_encoder.model.parameters(), 1.0)

    
    def validate_one_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Validate the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.ecg_encoder.eval()
        self.text_encoder.eval()
        self.shared_embedding_space.eval()
        self.contrastive_learning.eval()
        self.report_decoder.eval()

        total_contrastive_loss = 0.0
        total_ecg_captioning_loss = 0.0
        total_text_captioning_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)
        report_comparison = ""

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Load data and move to device
                ecg_waveforms = batch_data['waveform'].to(self.device)
                text_reports = batch_data['text']#.to(self.device)

                # Encode ECG and text into embeddings
                ecg_embeddings = self.ecg_encoder(ecg_waveforms)
                text_embeddings = self.text_encoder(text_reports)

                # Get shared embeddings using the shared embedding space
                ecg_shared, text_shared = self.shared_embedding_space(ecg_embeddings, text_embeddings)

                # Compute contrastive loss
                contrastive_loss = self.contrastive_learning(ecg_shared, text_shared)
                total_contrastive_loss += contrastive_loss.detach().item()

                #convert text_reports to tokens
                text_reports_tokenized = self.report_decoder.tokenizer(text_reports, return_tensors="pt", padding=True, truncation=True)
                text_reports_tokenized.to(next(self.report_decoder.parameters()).device) #send tokens to same device as decoder
                
                # Compute report generation loss using shared embeddings
                ecg_logits = self.report_decoder(shared_embedding=ecg_shared, target_text=text_reports_tokenized)
                ecg_captioning_loss = self.report_decoder.compute_captioning_loss(ecg_logits, target_text=text_reports_tokenized['input_ids'])
                text_logits = self.report_decoder(shared_embedding=text_shared, target_text=text_reports_tokenized)
                text_captioning_loss = self.report_decoder.compute_captioning_loss(text_logits, target_text=text_reports_tokenized['input_ids'])
                total_ecg_captioning_loss += ecg_captioning_loss.detach().item()
                total_text_captioning_loss += text_captioning_loss.detach().item()

                if batch_idx % 10 == 0:
                    text_example_report = self.report_decoder.generate_report(text_shared[0].unsqueeze(0))
                    ecg_example_report = self.report_decoder.generate_report(ecg_shared[0].unsqueeze(0))
                    
                    actual_report = f">>Actual report: \"{text_reports[0]}\"\n"
                    text_predicted_report = f">>Text predicted report: \"{text_example_report}\"\n"
                    ecg_predicted_report = f">>ECG predicted report: \"{ecg_example_report}\"\n"
                    print(actual_report)
                    print(text_predicted_report)
                    print(ecg_predicted_report)
                    report_comparison+=f">Batch: {batch_idx}"
                    report_comparison+=actual_report
                    report_comparison+=text_predicted_report
                    report_comparison+=ecg_predicted_report
        
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_ecg_captioning_loss = total_ecg_captioning_loss / num_batches
        avg_text_captioning_loss = total_text_captioning_loss / num_batches
        
        total_loss = total_contrastive_loss + total_ecg_captioning_loss + total_text_captioning_loss
        avg_total_loss = total_loss / num_batches
        loss_df = pd.DataFrame({
            "epoch": epoch,
            "avg_contrastive_loss": [avg_contrastive_loss],
            "avg_ecg_captioning_loss": [avg_ecg_captioning_loss],
            "avg_text_captioning_loss": [avg_text_captioning_loss],
            "avg_total_loss": [avg_total_loss]
        })
        return loss_df, report_comparison

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """
        Full training loop over multiple epochs.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
        """
        best_val_loss = float('inf')
        train_df = pd.DataFrame(columns=["epoch", "avg_contrastive_loss", "avg_captioning_loss", "avg_total_loss"])
        val_df = pd.DataFrame(columns=["epoch", "avg_contrastive_loss", "avg_captioning_loss", "avg_total_loss"])
        self.initial_llm_freeze()
        for epoch in range(self.training_params['num_epochs']):
            self.epoch = epoch
            print(f"Starting epoch {epoch + 1}/{self.training_params['num_epochs']}")            
            
            # Train for one epoch
            train_results_df, example_train_reports = self.train_one_epoch(train_dataloader, epoch)
            print(f"Tr Epoch [{epoch + 1}], AvgContL: {train_results_df['avg_contrastive_loss'][0]}, AvgECapL: {train_results_df['avg_ecg_captioning_loss'][0]}, AvgTCapL: {train_results_df['avg_text_captioning_loss'][0]}, AvgTotL: {train_results_df['avg_total_loss'][0]}")
            train_df = pd.concat([train_df, train_results_df], ignore_index=True)
            train_df.to_csv("train_results.csv")
            with open("train_example_reports.txt", 'a') as f:
                f.write(example_train_reports)
            # Validate for one epoch
            val_results_df, example_val_reports = self.validate_one_epoch(val_dataloader, epoch)
            print(f"Tr Epoch [{epoch + 1}], AvgContL: {val_results_df['avg_contrastive_loss'][0]}, AvgECapL: {val_results_df['avg_ecg_captioning_loss'][0]}, AvgTCapL: {val_results_df['avg_text_captioning_loss'][0]}, AvgTotL: {val_results_df['avg_total_loss'][0]}")
            val_df = pd.concat([val_df, val_results_df], ignore_index=True)
            val_df.to_csv("val_results.csv")
            with open("val_example_reports.txt", 'a') as f:
                f.write(example_val_reports)

            # Save model if validation loss improves
            if val_results_df['avg_total_loss'][0] < best_val_loss:
                best_val_loss = val_results_df['avg_total_loss'][0]
                print(f"Saving the best model with Validation Loss: {val_results_df['avg_total_loss'][0]}")
                self.save_checkpoint()

    def save_checkpoint(self, file_path: str = "best_model.pth"):
        """
        Save the model checkpoint to disk.

        Args:
            file_path (str, optional): Path to save the model checkpoint (default: "best_model.pth").
        """
        checkpoint = {
            'ecg_encoder_state_dict': self.ecg_encoder.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'shared_embedding_space_state_dict': self.shared_embedding_space.state_dict(),
            'contrastive_learning_state_dict': self.contrastive_learning.state_dict(),
            'report_decoder_state_dict': self.report_decoder.state_dict(),
            'optimizer_state_dict': {k: v.state_dict() for k, v in self.optimizer.items()}
        }
        torch.save(checkpoint, file_path)
        print(f"Model checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path: str):
        """
        Load the model checkpoint from disk.

        Args:
            file_path (str): Path to the model checkpoint file.
        """
        checkpoint = torch.load(file_path)
        self.ecg_encoder.load_state_dict(checkpoint['ecg_encoder_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.shared_embedding_space.load_state_dict(checkpoint['shared_embedding_space_state_dict'])
        self.contrastive_learning.load_state_dict(checkpoint['contrastive_learning_state_dict'])
        self.report_decoder.load_state_dict(checkpoint['report_decoder_state_dict'])
        for k, v in checkpoint['optimizer_state_dict'].items():
            self.optimizer[k].load_state_dict(v)
        print(f"Model checkpoint loaded from {file_path}")
