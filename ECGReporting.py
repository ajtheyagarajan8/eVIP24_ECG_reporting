from TrainingFramework import TrainingFramework
from ECGEncoder import ECGEncoder
from TextEncoder import TextEncoder
from SharedMetricSpace import SharedMetricSpace
from ContrastiveLearning import ContrastiveLearning
from ReportDecoder import ReportDecoder

from ECGDataLoader import ECGDataLoader, ECGDataBase
import warnings
warnings.filterwarnings("ignore")

#KEY ISSUES
# > CoCa produces sequence-aware embeddings and decodes this to text. We are attempting to decode a single (CLS) embedding, which is very ineffective

#TODO: actually use this
RANDOM_SEED = 42

TRAINING_PARAMS = {
            'num_epochs': 2,
            'batch_size': 8,
            'caption_loss_weight': 2,
            'contrastive_loss_weight': 1
        }
ECG_DATA_DIR_SUBSET_IMAGES = "mimic-iv-ecg_complete_300x300_images_to_p1067"
ECG_DATA_DIR_MINI_SUBSET_IMAGES = "mimic-iv-ecg_complete_300x300_images"

if __name__ == "__main__":
    #Instantiate database
    edb = ECGDataBase(ecg_data_dir=ECG_DATA_DIR_MINI_SUBSET_IMAGES)
    #uncomment to randomly remove N-100 samples from the database:
    #edb.random_undersample(100)
    
    #Subject-wise train val split 
    tr_subjects, vl_subjects = edb.train_val_split()
    print(f"tr_subjects: {len(tr_subjects)}")
    print(f"vl_subjects: {len(vl_subjects)}")

    #Instantiate ECGDataLoader objects to get a torch.utils.data.DataLoaders for train and validation sets
    tr_edl = ECGDataLoader(database=edb, split_subjects=tr_subjects, dynamic_loading=False, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
    tr_dl = tr_edl.get_dataloader()
    vl_edl = ECGDataLoader(database=edb, split_subjects=vl_subjects, dynamic_loading=False, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
    vl_dl = vl_edl.get_dataloader()

    #Instantiate model components 
    text_encoder = TextEncoder(pretrained_model="michiyasunaga/BioLinkBERT-base")
    report_decoder = ReportDecoder(decoder_name = 'biogpt')
    rd_emb_dim = report_decoder.decoder_input_dimension
    ecg_encoder = ECGEncoder(model_architecture="ResNet101", representation_embedding_dim=rd_emb_dim, pretrained_model="resnetPTBXL_weights.pth")
    te_emb_dim = text_encoder.embedding_dim
    shared_metric_space = SharedMetricSpace(ecg_embedding_dim=ecg_encoder.representation_embedding_dim, text_embedding_dim=text_encoder.embedding_dim, shared_metric_embedding_dim=text_encoder.embedding_dim)
    contrastive_learning = ContrastiveLearning()

    tfw = TrainingFramework(ecg_encoder, text_encoder, shared_metric_space, contrastive_learning, report_decoder, TRAINING_PARAMS)

    #tfw.load_checkpoint("best_model.pth")

    tfw.train(tr_dl, vl_dl)





