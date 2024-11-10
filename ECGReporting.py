from TrainingFramework import TrainingFramework
from ECGEncoder import ECGEncoder
from TextEncoder import TextEncoder
from SharedEmbeddingSpace import SharedEmbeddingSpace
from ContrastiveLearning import ContrastiveLearning
from ReportDecoder import ReportDecoder

from ECGDataLoader import ECGDataLoader, TrainValSplit

RANDOM_SEED = 42

if __name__ == "__main__":
    ecg_encoder = ECGEncoder()
    text_encoder = TextEncoder()
    shared_embedding_space = SharedEmbeddingSpace()
    contrastive_learning = ContrastiveLearning()
    report_decoder = ReportDecoder()
    
    tvs = TrainValSplit()
    tr_subjects, vl_subjects = tvs.train_val_split()

    ecg_dl_tr = ECGDataLoader(tr_subjects)
    training_dataloader = ecg_dl_tr.get_dataloader()

    ecg_dl_vl = ECGDataLoader(vl_subjects)
    validating_dataloader = ecg_dl_vl.get_dataloader()

    tfw = TrainingFramework(ecg_encoder, text_encoder, shared_embedding_space, contrastive_learning, report_decoder)

    tfw.train(training_dataloader, validating_dataloader)





