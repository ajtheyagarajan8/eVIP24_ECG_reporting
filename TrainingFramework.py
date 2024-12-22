import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

class TrainingFramework:
    """
    TrainingFramework class for managing the end-to-end training and validation process.

    Attributes:
        ecg_encoder (nn.Module): Instance of the ECGEncoder module.
        text_encoder (nn.Module): Instance of the TextEncoder module.
        shared_metric_space (nn.Module): Instance of the SharedMetricSpace module.
        contrastive_learning (nn.Module): Instance of the ContrastiveLearning module.
        report_decoder (nn.Module): Instance of the ReportDecoder module.
        device (torch.device): Device to run the training on (CPU or GPU).
        training_params (dict): Dictionary containing training parameters (e.g., learning rate, batch size).
    """

    def __init__(self, 
                 ecg_encoder: nn.Module, 
                 text_encoder: nn.Module, 
                 shared_metric_space: nn.Module, 
                 contrastive_learning: nn.Module, 
                 report_decoder: nn.Module,
                 training_params: dict = None):
        """
        Initialize the TrainingFramework.

        Args:
            ecg_encoder (nn.Module): Initialized ECGEncoder module.
            text_encoder (nn.Module): Initialized TextEncoder module.
            shared_metric_space (nn.Module): Initialized SharedMetricSpace module.
            contrastive_learning (nn.Module): Initialized ContrastiveLearning module.
            report_decoder (nn.Module): Initialized ReportDecoder module.
            training_params (dict, optional): Dictionary containing training parameters (default: None).
        """
        self.epoch = 0
        self.batch_idx = 0
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder
        self.shared_metric_space = shared_metric_space
        self.contrastive_learning = contrastive_learning
        self.report_decoder = report_decoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache() 
        self.ecg_encoder.to(self.device)
        self.text_encoder.to(self.device)
        self.shared_metric_space.to(self.device)
        self.report_decoder.to(self.device)
        # Set default training parameters if not provided
        self.ecg_encoder_params = {'learning_rate': 1e-4, 'weight_decay':1e-3}
        self.text_encoder_params = {'learning_rate':1e-4, 'weight_decay': 1e-5}
        self.shared_optim_params = {'learning_rate': 1e-4, 'weight_decay':1e-3}
        self.report_dencoder_params = {'learning_rate':1e-4, 'weight_decay': 1e-5}
        self.training_params = training_params

        # Optimizers for each component
        self.optimizer = self.initialize_optimizers()

        self.scheduler = {
            'ecg_scheduler': lr_scheduler.StepLR(self.optimizer['ecg_optimizer'], step_size=5, gamma=0.9),
            'text_scheduler': lr_scheduler.StepLR(self.optimizer['text_optimizer'], step_size=5, gamma=0.9),
            'shared_metric_space_scheduler': lr_scheduler.StepLR(self.optimizer['shared_metric_space_optimizer'], step_size=10, gamma=0.1),
            'report_decoder_scheduler': lr_scheduler.StepLR(self.optimizer['report_decoder_optimizer'], step_size=5, gamma=0.9)
        }

    def initialize_optimizers(self):
        """
        Initialize optimizers for each module in the framework.

        Returns:
            dict: Dictionary of optimizers for each module.
        """
        #TODO: would it be best to only register non-frozen weights with the optimizer? Would this save compute?
        ecg_optimizer = optim.Adam(self.ecg_encoder.parameters(), lr=self.ecg_encoder_params['learning_rate'], weight_decay=self.ecg_encoder_params['weight_decay'])
        text_optimizer = optim.Adam(self.text_encoder.parameters(), lr=self.text_encoder_params['learning_rate'], weight_decay=self.text_encoder_params['weight_decay'])
        shared_metric_space_optimizer = optim.Adam(self.shared_metric_space.parameters(), lr=self.shared_optim_params['learning_rate'], weight_decay=self.shared_optim_params['weight_decay'])
        report_decoder_optimizer = optim.Adam(self.report_decoder.parameters(), lr=self.report_dencoder_params['learning_rate'], weight_decay=self.report_dencoder_params['weight_decay'])

        return {
            'ecg_optimizer': ecg_optimizer,
            'text_optimizer': text_optimizer,
            'shared_metric_space_optimizer': shared_metric_space_optimizer,
            'report_decoder_optimizer': report_decoder_optimizer
        }

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.

        Returns:
            float: Average loss for the epoch.
        """
        self.ecg_encoder.train()
        self.text_encoder.train()
        self.shared_metric_space.train()
        self.contrastive_learning.train()
        self.report_decoder.train()

        total_contrastive_loss = 0.0
        total_captioning_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)
        report_comparison = ""
        total_bleu_score = 0.0
        results_freq = 10 #how many batches until we compute evaluation metrics, save results, print sample generated report results
        save_freq = 100
        lr_scheduler_freq = 100

        #best_avg_ecg_bleu_score_from_results_big_epoch1 = 0.00581593002820958
        #best_avg_txt_bleu_score_from_results_big_epoch1 = 0.00519173327995373
        #best_avg_bleu = (best_avg_ecg_bleu_score_from_results_big_epoch1 + best_avg_txt_bleu_score_from_results_big_epoch1)/2
        best_bleu = 0.0

        train_df = pd.DataFrame(columns=["epoch", "batch", "avg_contrastive_loss", "avg_captioning_loss", "avg_total_loss", "avg_bleu_score"])

        for batch_idx, batch_data in enumerate(dataloader):
            # Load data and move to device
            ecg_waveforms = batch_data['waveform'].to(self.device)
            text_reports = batch_data['text']
            # Encode ECG and text into embeddings
            ecg_rep_embeddings = self.ecg_encoder(ecg_waveforms)
            text_rep_embeddings = self.text_encoder(text_reports)
            # Get shared embeddings using the shared embedding space
            ecg_shared_metric, text_shared_metric = self.shared_metric_space(ecg_rep_embeddings, text_rep_embeddings)
            # Compute contrastive loss from text_shared
            contrastive_loss = self.contrastive_learning(ecg_shared_metric, text_shared_metric)
            #convert text_reports to tokens
            text_reports_tokenized = self.report_decoder.tokenizer(text_reports, return_tensors="pt", padding=True, truncation=True)
            text_reports_tokenized.to(self.report_decoder.curr_device())
            #decode text report token logits from ecg_rep_embeddings
            report_logits = self.report_decoder(ecg_rep_embeddings, text_reports_tokenized)
            report_captioning_loss = self.report_decoder.compute_captioning_loss(report_logits, text_reports_tokenized)
            
            # Combine losses for back prop
            total_loss_value = self.training_params['contrastive_loss_weight']*contrastive_loss + self.training_params['caption_loss_weight']*report_captioning_loss

            # Backpropagate and optimize each component
            self.optimizer['ecg_optimizer'].zero_grad()
            self.optimizer['text_optimizer'].zero_grad()
            self.optimizer['shared_metric_space_optimizer'].zero_grad()
            self.optimizer['report_decoder_optimizer'].zero_grad()

            total_loss_value.backward()
            self.progressive_unfreeze()
            self.apply_gradient_clipping()

            self.optimizer['ecg_optimizer'].step()
            self.optimizer['text_optimizer'].step()
            self.optimizer['shared_metric_space_optimizer'].step()
            self.optimizer['report_decoder_optimizer'].step()

            # Accumulate loss for results
            total_loss += total_loss_value.detach().item()
            total_contrastive_loss += contrastive_loss.detach().item()
            total_captioning_loss += report_captioning_loss.detach().item()

            print(f"Epoch [{self.epoch + 1}/{self.training_params['num_epochs']}], Batch [{batch_idx + 1}/{num_batches}], avg_total_loss: {total_loss / num_batches}, avg_captioning_loss: {total_captioning_loss / num_batches}, avg_contrastive_loss: {total_contrastive_loss / num_batches}")
            if batch_idx % results_freq == 0:
                #print example text for one sample in the batch and compute evaluation metrics
                generated_report = self.report_decoder.generate_report(ecg_rep_embeddings[0].unsqueeze(0).unsqueeze(1))
                # Compute BLEU score
                #TODO: compute BLEU this for the whole batch not just one sample
                actual_report = text_reports[0]
                reference = [actual_report.split()]
                candidate = generated_report.split()
                bleu_score = sentence_bleu(reference, candidate)
                total_bleu_score += bleu_score
                
                actual_report = f"\n>>Actual report: \"{actual_report}\"\n"
                predicted_report = f">>Predicted report: \"{generated_report}\"\n"
                bleu_score_report = f">>>BLEU score: {bleu_score}\n"
                print(actual_report)
                print(predicted_report)
                print(bleu_score_report)
                report_comparison+=f">Batch: {batch_idx}\n"
                report_comparison+=actual_report
                report_comparison+=predicted_report
                report_comparison+=bleu_score_report
                
                curr_loss_df = pd.DataFrame({
                    "epoch": self.epoch,
                    "batch": batch_idx+1,
                    "avg_contrastive_loss": [total_contrastive_loss / (batch_idx+1)],
                    "avg_captioning_loss": [total_captioning_loss / (batch_idx+1)],
                    "avg_total_loss": [total_loss / (batch_idx+1)],
                    "avg_bleu_score": [total_bleu_score / ((batch_idx+1) // results_freq + 1)],
                })
                train_df = pd.concat([train_df, curr_loss_df], ignore_index=True)
                #curr_loss_df.to_csv("train_batch_results.csv")
                with open("train_batch_example_reports.txt", 'a') as f:
                    f.write(report_comparison)
                report_comparison = ""
                    
            if batch_idx != 0 and batch_idx % lr_scheduler_freq == 0:
                print("Updating LR scheduler")
                for scheduler in self.scheduler.values():
                    scheduler.step()
            #if batch_idx % save_freq == 0:
            #    print("Saving current model")
            #    self.save_checkpoint()
        
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_ecg_captioning_loss = total_captioning_loss / num_batches
        avg_total_loss = total_loss / num_batches
        avg_bleu_score = total_bleu_score / (num_batches // results_freq + 1)
        loss_df = pd.DataFrame({
            "epoch": self.epoch,
            "avg_contrastive_loss": [avg_contrastive_loss],
            "avg_captioning_loss": [avg_ecg_captioning_loss],
            "avg_total_loss": [avg_total_loss],
            "avg_bleu_score": [avg_bleu_score]
        })
        return loss_df

    def freeze_pretrained(self):
        self.text_encoder.freeze_encoder()
        self.report_decoder.freeze_decoder()
        self.ecg_encoder.freeze_encoder(freeze_metric_embedding=False)

    def progressive_unfreeze(self):
        if self.epoch <= 2:
            # Unfreeze top 2 layers of text encoder and report decoder
            for param in self.text_encoder.model.encoder.layer[-2:].parameters():
                param.requires_grad = True  
            for param in self.report_decoder.decoder.biogpt.layers[-2:].parameters():
                param.requires_grad = True
        elif self.epoch > 2 and self.epoch <= 5:
            # Unfreeze top 4 layers of text encoder and report decoder
            for param in self.text_encoder.model.encoder.layer[-4:].parameters():
                param.requires_grad = True
            for param in self.report_decoder.decoder.biogpt.layers[-4:].parameters():
                param.requires_grad = True

    def apply_gradient_clipping(self):
        torch.nn.utils.clip_grad_norm(self.report_decoder.decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm(self.text_encoder.model.parameters(), 1.0)
    
    def validate_one_epoch(self, dataloader: DataLoader) -> float:
        """
        Validate the model for one epoch.

        Args:
            dataloader (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.ecg_encoder.eval()
        self.text_encoder.eval()
        self.shared_metric_space.eval()
        self.contrastive_learning.eval()
        self.report_decoder.eval()

        total_contrastive_loss = 0.0
        total_captioning_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)
        report_comparison = ""
        total_bleu_score = 0.0
        results_freq = 10

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                ecg_waveforms = batch_data['waveform'].to(self.device)
                text_reports = batch_data['text']
                # Encode ECG and text into embeddings
                text_rep_embeddings = self.text_encoder(text_reports)
                ecg_rep_embeddings = self.ecg_encoder(ecg_waveforms)
                # Get shared embeddings using the shared embedding space
                ecg_shared_metric, text_shared_metric = self.shared_metric_space(ecg_rep_embeddings, text_rep_embeddings)
                # Compute contrastive loss from text_shared
                contrastive_loss = self.contrastive_learning(ecg_shared_metric, text_shared_metric)
                #convert text_reports to tokens
                text_reports_tokenized = self.report_decoder.tokenizer(text_reports, return_tensors="pt", padding=True, truncation=True)
                text_reports_tokenized.to(self.report_decoder.curr_device())
                # Compute report generation loss using shared embeddings
                report_logits = self.report_decoder(ecg_rep_embeddings, text_reports_tokenized)
                report_captioning_loss = self.report_decoder.compute_captioning_loss(report_logits, text_reports_tokenized)
                
                total_captioning_loss += report_captioning_loss.detach().item()
                total_loss_value = self.training_params['contrastive_loss_weight']*contrastive_loss + self.training_params['caption_loss_weight']*report_captioning_loss
                total_loss += total_loss_value.detach().item()

                if batch_idx % results_freq == 0:
                    generated_report = self.report_decoder.generate_report(ecg_rep_embeddings[0].unsqueeze(0).unsqueeze(1))
                    # Compute BLEU score
                    #TODO: compute BLEU this for the whole batch not just one sample
                    actual_report = text_reports[0]
                    reference = [actual_report.split()]
                    candidate = generated_report.split()
                    bleu_score = sentence_bleu(reference, candidate)
                    total_bleu_score += bleu_score
                    
                    actual_report = f">>Actual report: \"{actual_report}\"\n"
                    predicted_report = f">>Predicted report: \"{generated_report}\"\n"
                    bleu_score_report = f">>>BLEU score: {bleu_score}\n"
                    print(actual_report)
                    print(predicted_report)
                    print(bleu_score_report)
                    report_comparison+=f">Batch: {batch_idx}"
                    report_comparison+=actual_report
                    report_comparison+=predicted_report
                    report_comparison+=bleu_score_report
        
        avg_captioning_loss = total_captioning_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_bleu_score = total_bleu_score / (num_batches // results_freq + 1)
        avg_total_loss = total_loss / num_batches
        
        loss_df = pd.DataFrame({
            "epoch": self.epoch,
            "avg_contrastive_loss": [avg_contrastive_loss],
            "avg_captioning_loss": [avg_captioning_loss],
            "avg_total_loss": [avg_total_loss],
            "avg_bleu_score": [avg_bleu_score]
        })
        return loss_df

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """
        Full training loop over multiple epochs.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
        """
        best_val_loss = float('inf')
        train_df = pd.DataFrame(columns=["epoch", "avg_contrastive_loss", "avg_captioning_loss", "avg_total_loss", "avg_bleu_score"])
        val_df = pd.DataFrame(columns=["epoch", "avg_contrastive_loss", "avg_captioning_loss", "avg_total_loss", "avg_bleu_score"])
        self.freeze_pretrained()
        for epoch in range(self.training_params['num_epochs']):
            self.epoch = epoch
            print(f"Starting epoch {epoch + 1}/{self.training_params['num_epochs']}")            
            
            # Train for one epoch
            train_results_df = self.train_one_epoch(train_dataloader)
            print(f"Tr Epoch [{epoch + 1}], avg_bleu_score: {train_results_df['avg_bleu_score'][0]}, avg_total_loss: {train_results_df['avg_total_loss'][0]}, avg_captioning_loss: {train_results_df['avg_captioning_loss'][0]}, avg_contrastive_loss: {train_results_df['avg_contrastive_loss'][0]}")
            train_df = pd.concat([train_df, train_results_df], ignore_index=True)
            train_df.to_csv("train_results.csv")
            
            # Validate for one epoch
            val_results_df = self.validate_one_epoch(val_dataloader)
            print(f"Tr Epoch [{epoch + 1}], avg_bleu_score: {val_results_df['avg_bleu_score'][0]}, avg_total_loss: {val_results_df['avg_total_loss'][0]}, avg_captioning_loss: {val_results_df['avg_captioning_loss'][0]}, avg_contrastive_loss: {val_results_df['avg_contrastive_loss'][0]}")
            val_df = pd.concat([val_df, val_results_df], ignore_index=True)
            val_df.to_csv("val_results.csv")

            #Save model if validation loss improves
            if val_df['avg_total_loss'][0] < best_val_loss:
                best_val_loss = train_df['avg_total_loss'][0]
                print(f"Saving the best model with validation Loss: {train_df['avg_total_loss'][0]}")
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
            'shared_metric_space_state_dict': self.shared_metric_space.state_dict(),
            'contrastive_learning_state_dict': self.contrastive_learning.state_dict(),
            'report_decoder_state_dict': self.report_decoder.state_dict(),
            'optimizer_state_dict': {k: v.state_dict() for k, v in self.optimizer.items()},
            'scheduler_state_dict': {k: v.state_dict() for k, v in self.scheduler.items()}
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
        self.shared_metric_space.load_state_dict(checkpoint['shared_metric_space_state_dict'])
        self.contrastive_learning.load_state_dict(checkpoint['contrastive_learning_state_dict'])
        self.report_decoder.load_state_dict(checkpoint['report_decoder_state_dict'])
        for k, v in checkpoint['optimizer_state_dict'].items():
            self.optimizer[k].load_state_dict(v)
        for k, v in checkpoint['scheduler_state_dict'].items():
            self.scheduler[k].load_state_dict(v)
        print(f"Model checkpoint loaded from {file_path}")

    def train_one_epoch_old(self, dataloader: DataLoader, epoch: int) -> float:
        #NOT USED ANYMORE
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
        self.shared_metric_space.train()
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
            self.batch_idx = batch_idx + 1
            ecg_waveforms = batch_data['waveform'].to(self.device)
            text_reports = batch_data['text']#.to(self.device)

            # Encode ECG and text into embeddings
            ecg_embeddings = self.ecg_encoder(ecg_waveforms)
            text_embeddings = self.text_encoder(text_reports)
            # Get shared embeddings using the shared embedding space
            ecg_shared, text_shared = self.shared_metric_space(ecg_embeddings, text_embeddings)
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
            self.optimizer['shared_metric_space_optimizer'].zero_grad()
            self.optimizer['report_decoder_optimizer'].zero_grad()

            total_loss_value.backward()
            #self.progressive_unfreeze()
            #self.apply_gradient_clipping()

            self.optimizer['ecg_optimizer'].step()
            self.optimizer['text_optimizer'].step()
            self.optimizer['shared_metric_space_optimizer'].step()
            self.optimizer['report_decoder_optimizer'].step()

            # Accumulate loss for reporting
            total_loss += total_loss_value.detach().item()
            print(f"Epoch [{epoch + 1}/{self.training_params['num_epochs']}], Batch [{batch_idx + 1}/{num_batches}], ContLoss: {contrastive_loss.item()}, ECGCapLoss: {ecg_captioning_loss.item()}, TxtCapLoss: {text_captioning_loss.item()}, Loss: {total_loss_value.item()}")
            if batch_idx % 5 == 0:
                    text_example_report = self.report_decoder.generate_report(text_shared[0].unsqueeze(0))
                    ecg_example_report = self.report_decoder.generate_report(ecg_shared[0].unsqueeze(0))
                    
                    actual_report = f"\n>>Actual report: \"{text_reports[0]}\"\n"
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

# def init_live_results_plot(self):
    #     self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 9))
    #     fig_title = fig.suptitle("Training Results")
    #     axs = list(chain(*self.axes))
    #     axs[0].plot([],[], lw=1, color="green", label="Contrastive")


    # def update_live_results_plot(self, contrastive_loss, ecg_cap_loss, text_cap_loss, ecg_bleu=None, text_bleu=None, ):
        

    #     # Create a blank line. We will update the line in animate
    #     lines = []
    #     pos = 0
    #     device_iterator_index = 0
    #     channel_iterator_index = 0
    #     channels_per_device = len(plot_configs)
    #     total_number_of_traces = channels_per_device * device_number

    #     # Axes are plotted according to their listed position in settings.json
    #     for ax in axs:
    #         # Turn off any axes after the last plot
    #         # if pos >= len(plot_configs):
    #         if pos >= total_number_of_traces:
    #             ax.set_axis_off()
    #             continue

    #         curr_sensor = plot_configs[channel_iterator_index]

    #         # Get each line in the axis
    #         for i in range(curr_sensor["channelNumber"]):
    #             ax.set_title(curr_sensor["name"])
    #             # TODO: X and Y labels?
    #             ax_line, = ax.plot([], [],
    #                             lw=1,
    #                             color=curr_sensor["plot"]["colour"][i],
    #                             label=curr_sensor["channelNames"][i])
    #             lines.append(ax_line)

    #         # Set up current axis
    #         ax.legend(loc="upper left")
    #         ax.set_ylim(curr_sensor["plot"]["ylim"])
    #         ax.set_xlim([id_start, id_end])
    #         ax.autoscale(enable=True, axis="y", tight=True)
    #         ax.grid()

    #         channel_iterator_index += 1
    #         if channel_iterator_index == channels_per_device:
    #             channel_iterator_index = 0
    #             device_iterator_index += 1
    #         pos += 1
    #     ani = animation.FuncAnimation(fig,
    #                               data_update,
    #                               fargs=(data_q, log_q, lines, axs, plot_configs, trace_info, device_number),
    #                               interval=10,
    #                               blit=True)