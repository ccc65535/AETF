import torch
import sys

sys.path.append(r"../braindecode")
from Mixup import BGMix
from braindecode.augmentation import FrequencyShift # type: ignore
from braindecode.augmentation import FTSurrogate    # type: ignore
from braindecode.augmentation import GaussianNoise  # type: ignore
from braindecode.augmentation import IdentityTransform  # type: ignore
from braindecode.augmentation import SignFlip   # type: ignore
from braindecode.augmentation import SmoothTimeMask # type: ignore
from braindecode.augmentation import TimeReverse    # type: ignore
from braindecode.augmentation import ChannelsShuffle    # type: ignore



def augment(trans_type,N,X,y,srate=250):

    aug_X=torch.zeros(X.shape)
    aug_y=torch.zeros(y.shape)
    if trans_type=='SignalFlip':
        transform = SignFlip(
            probability=.5,
        )

    elif trans_type=='TimeReverse':
        transform = TimeReverse(
            probability=.5,
        )

    elif trans_type=='SmoothTimeMask':
        transform = SmoothTimeMask(
            probability=.5,
            mask_len_samples=int(X.shape[-1]/2)
        )

    elif trans_type=='GaussianNoise':
        transform = GaussianNoise(
            probability=.5,
        )

    elif trans_type=='FrequencyShift':
        transform = FrequencyShift(
            probability=.5,
            sfreq=srate
        )

    elif trans_type=='FTSurrogate':
        transform = FTSurrogate(
            probability=.5,
        )

    elif trans_type=='ChannelsShuffle':
        transform = ChannelsShuffle(
            probability=.5,
            p_shuffle=0.8
        )

    # elif trans_type=='BGMix':
    #     aug_X,aug_y=BGMix.generate_data(X.numpy(),y,nt=[2 for i in range(N)]+[3 for i in range(N)])
    #     return aug_X,aug_y


    else:
        assert('aug method not founded!')
    # rng=np.random.randint(1,100,N)
    for i in range(N):
        # transform.rng=check_random_state(rng[i])
        X_tr, y_tr = transform(X,y)
        X_tr
        if i==0:
            aug_X=X_tr
            aug_y=y_tr
        else:
            aug_X=torch.cat((aug_X,X_tr),dim=0)
            aug_y=torch.cat((aug_y,y_tr),dim=0)

    return aug_X,aug_y