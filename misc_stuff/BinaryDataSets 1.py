import ecg_plot
import pandas as pd
import numpy as np
import wfdb
import ast
import os
from PIL import Image

# Preprocessing of MINet Datasets, organising normal and MI ecgs into seperate folders

# Read in data from ptb-xl dataset

# Read Labels, focusing only on NORM ECGs and MI ecgs
# Perform processing on ecgs to convert to images and at 300px size
# Then place them in designated label folder

diagnostic_normal = 'NORM'
diagnostic_target = 'AFLT'

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


print('Pre-loading database...')
cwd = os.getcwd()
path = cwd + '/ptbxl/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

print('Database loaded!')
print('Performing digital to paper ECG conversion...')

agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

net_dir= '/home/han/Documents/George_Hill_Lab/VC_ECG_Analysis/'+diagnostic_target+'Net/'
try:
    os.mkdir(net_dir)
except:
    pass
dataset_dir = net_dir + diagnostic_target+'NetImages/'
try:
    os.mkdir(dataset_dir)
except:
    pass

save_path = dataset_dir + 'ECGIMAGES' + '/'
try:
    os.mkdir(save_path)
except:
    pass

net_input_size = 300

# Tally how many labels for each there are:
labels = []
filenumber = 0
norms = 0
targs = 0


for index in range(len(X)):
    # diagnosis = Y.diagnostic_superclass.values[index]
    diagnosis = Y.scp_codes.values[index]
    if diagnostic_normal in diagnosis:
        labels.append(0)
        norms += 1
        # save_path = (dataset_dir + '/' + diagnostic_normal + '/')
    elif diagnostic_target in str(diagnosis):
        labels.append(1)
        targs += 1
        # save_path = (dataset_dir + '/' + diagnostic_target + '/')
    else:   
        continue
    ecg = np.transpose(X[index,:,:])
    # plt = ecg_plot.plot_1(ecg, sample_rate = 500, title = 'ECG 12', fig_height=3.125, fig_width=3.125)
    plt = ecg_plot.plot(ecg, sample_rate = 500, title = ' ', style = 'bw')
    ecg_plot.save_as_png('temp',dataset_dir)
    image = Image.open(dataset_dir + 'temp.png')
    squaredImage = expand2square(image, (0,0,0))
    newImage = squaredImage.resize((net_input_size,net_input_size))
    newImage.save(save_path + str(filenumber) + '.png')
    filenumber += 1
    print(str(index) + ' out of ' + str(len(X)))


np.save(dataset_dir + 'labels.npy', labels)

print(str(norms) + ' normals and ' + str(targs) + ' ' +diagnostic_target + 's')















'''
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from transformers import AutoTokenizer, AutoModelForCausalLM

class ECGReport(nn.Module):
    def __init__(self):
        #define ECG encoder
        self.ResNet101 = models.resnet101(pretrained=True)
        self.ResNet101.fc = nn.Linear(ResNet101.fc.in_features, 512)
        model_dict = ResNet101.state_dict()
        pretrained_dict = torch.load('resnetPTBXL_weights.pth')
        # Filter out the final fc layer weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        # Update the model's state dict with the filtered weights
        model_dict.update(pretrained_dict)
        ResNet101.load_state_dict(model_dict)
        self.ecg_encoder = ResNet101
        #define text encoder
        self.text_encoder = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
        self.text_tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
        # Optional: Project output to the specified embedding dimension if needed
        self.text_embedding_projection = nn.Linear(self.text_encoder.config.hidden_size, 512)
        #shared embedding space
        self.ecg_projection_layer = nn.Linear(512, 512)
        self.text_projection_layer = nn.Linear(512, 512)
        #report decoder
        self.report_tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        self.report_decoder = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
        self.vocab_size = self.decoder.config.vocab_size
        self.decoder_input_size = self.decoder.config.hidden_size
        self.shared_embedding_projection = nn.Linear(512, self.decoder_input_size)
    
    def forward(self, ecg_input, text_input):
        ecg_embeddings = self.ecg_encoder(ecg_input)
        ecg_shared = self.ecg_projection_layer(ecg_embeddings)
        shared_embeddings_list = [ecg_shared]

        if text_input is not None:
            text_token_inputs = self.text_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                text_outputs = self.text_encoder(**text_token_inputs)
            # Get the last hidden state (or use [CLS] token embedding)
            last_hidden_state = text_outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
            text_embeddings = self.text_embedding_projection(last_hidden_state[:, 0, :])  # Shape: (batch_size, embedding_dim)
            text_shared = self.text_projection_layer(text_embeddings)
            shared_embeddings_list.append(text_shared)
        
        report_logits = []
        for shared_embedding in shared_embeddings_list:
            decoder_input = self.embedding_projection(shared_embedding)
            if text_input is not None:
                target_text_tokenized = self.text_encoder.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
                batch_size = shared_embedding.size(0)
                seq_length = target_text_tokenized['input_ids'].size(1)
                decoder_input = decoder_input.unsqueeze(1).expand(-1, seq_length, -1)  # Shape: (batch_size, seq_length, n_embd)
                outputs = self.decoder(inputs_embeds=decoder_input, labels=target_text_tokenized['input_ids'])
            else:
                outputs = self.decoder(inputs_embeds=decoder_input)
            report_logits.append(outputs.logits)  # Shape: (batch_size, seq_length, vocab_size)
        
        ecg_report_logits = report_logits[0]
        if len(report_logits)>1:
            text_report_logits = report_logits[1]
        else:
            text_report_logits = None

        return ecg_shared, text_shared, ecg_report_logits, text_report_logits

def train_ecg_report(dataloader):
    ecgr = ECGReport()
    optimizer = optim.Adam(self.ecgr.parameters(), lr=1e-4, weight_decay=1e-5)
    ecgr.train()
    for _, batch_data in enumerate(dataloader):
        ecg_waveforms = batch_data['waveform']
        text_reports = batch_data['text']
        ecg_shared, text_shared, ecg_report_logits, text_report_logits = ecgr(ecg_waveforms, text_reports)


        ecg_normalized = F.normalize(ecg_shared, p=2, dim=-1)
        text_normalized = F.normalize(text_shared, p=2, dim=-1)
        similarity_scores = torch.matmul(ecg_normalized, text_normalized.T)  # Shape: (batch_size, batch_size)
        batch_size = ecg_embedding.size(0)
        temperature = 0.07
        scaled_similarity = similarity_scores / temperature
        labels = torch.arange(batch_size).long().to(scaled_similarity.device) #Shape (batch_size,)
        loss_fn = nn.CrossEntropyLoss()
        contrastive_loss = loss_fn(scaled_similarity, labels)
        
        text_reports_tokenized = ecgr.text_tokenizer(text_reports, return_tensors="pt", padding=True, truncation=True)
        def captioning_loss(logits, target_text):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_target_text = target_text[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            captioning_loss = loss_fn(shift_logits.view(-1, self.vocab_size), shift_target_text.view(-1))
            return captioning_loss

        ecg_captioning_loss = captioning_loss(ecg_report_logits, target_text=text_reports_tokenized['input_ids'])
        text_captioning_loss = captioning_loss(text_report_logits, target_text=text_reports_tokenized['input_ids'])
        total_loss = contrastive_loss + captioning_loss
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()
'''