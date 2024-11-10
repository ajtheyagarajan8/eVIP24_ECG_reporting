from TrainingFramework import TrainingFramework
from ECGEncoder import ECGEncoder
from TextEncoder import TextEncoder
from SharedEmbeddingSpace import SharedEmbeddingSpace
from ContrastiveLearning import ContrastiveLearning
from ReportDecoder import ReportDecoder

from ECGDataLoader import ECGDataLoader

if __name__ == "__main__":
    ecg_encoder = ECGEncoder()
    text_encoder = TextEncoder()
    shared_embedding_space = SharedEmbeddingSpace()
    contrastive_learning = ContrastiveLearning()
    report_decoder = ReportDecoder()
    
    
    data_path_train = ""
    ecg_dl_tr = ECGDataLoader()
    training_dataloader = ecg_dl_tr.get_dataloader()

    data_path_val = ""
    ecg_dl_vl = ECGDataLoader()
    validating_dataloader = ecg_dl_vl.get_dataloader()

    tfw = TrainingFramework(ecg_encoder, text_encoder, shared_embedding_space, contrastive_learning, report_decoder)

    tfw.train(training_dataloader, validating_dataloader)





