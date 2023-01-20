

from braindecode import EEGClassifier
from sklearn.model_selection import train_test_split

from braindecode.datasets import SleepPhysionet

from sklearn.preprocessing import robust_scale
from braindecode.preprocessing import create_windows_from_events
import numpy as np
from sklearn.utils import compute_class_weight
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring, LRScheduler
from sklearn.metrics import balanced_accuracy_score
import torch
from braindecode.util import set_random_seeds
from woods.models import BENDR



if __name__ == "__main__":
    subject_ids = [0]

    dataset = SleepPhysionet(
        subject_ids=subject_ids, recording_ids=None, crop_wake_mins=30)



    # preprocessors = [Preprocessor(robust_scale, channel_wise=True)]
    #
    # preprocess(dataset, preprocessors)



    mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    window_size_s = 16
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=30*100,
        preload=True,
        mapping=mapping,
    )


    split_ids = dict(train=subject_ids[0:2],
                     test=subject_ids[3:4] ,
                     valid = [5])

    splits = windows_dataset.split(split_ids)
    train_set, test_set, valid_set = splits["train"], splits["test"], splits["valid"]


    ##############################################

    model_hparams = {
        'model': 'BENDR',
        'encoder_h': 512,
        'projection_head': False,
        'enc_do': 0.1,
        'feat_do': 0.4,
        'pool_length': 4,
        'mask_p_t': 0.01,
        'mask_p_c': 0.005,
        'mask_t_span': 0.05,
        'mask_c_span': 0.1,
        'classifier_layers': 1,
        'model_path': None
    }
#input = torch.empty(2, 3000)
#data = torch.zeros_like(input)
#resul = model.channel_embedding(data)
    from skorch.helper import SliceDataset

    model = BENDR(windows_input=1600, samples=100,
                  original_channel_size =2,
                  model_hparams=model_hparams)

    x = torch.from_numpy(np.array(SliceDataset(windows_dataset, 0)))
    model(x[::, :1])
    x.shape#([2254, 2, 1600])
    #testando com todas as janelas
    result_em = model.channel_embedding(x)
    result_em.shape#([2254, 20, 2, 1599])
    #testando com uma janela
    result_em_2 = model.channel_embedding(x[::, :1])

    result_em_2.shape#([2254, 20, 1599])

    result_en = model.encoder(result_em_2)
    result_en.shape#([2254, 512, 17])

    result_aug = model.enc_augment(result_en)
    result_aug.shape#([2254, 1536, 17])

    result_sum = model.summarizer(result_aug)
    result_sum.shape#([2254, 1536, 4])

    result_features = model.extended_classifier(result_sum)
    result_features.shape#([2254, 2048])

    result_log = model.classifier(result_features)
    result_log.shape#([2254, 1600])

    result = result_log.unsqueeze(1)
    result.shape#([2254, 1, 1600])
    lr = 1e-3
    batch_size = 32
    n_epochs = 5


    cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True

    set_random_seeds(seed=31, cuda=cuda)



    def balanced_accuracy_multi(model, X, y):
        y_pred = model.predict(X)
        return balanced_accuracy_score(y.flatten(), y_pred.flatten())


    train_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi,
        on_train=True,
        name='train_bal_acc',
        lower_is_better=False,
    )
    valid_bal_acc = EpochScoring(
        scoring=balanced_accuracy_multi,
        on_train=False,
        name='valid_bal_acc',
        lower_is_better=False,
    )
    callbacks = [
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc)
    ]
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 4

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )


    clf.fit(train_set, y=None, epochs=n_epochs)





