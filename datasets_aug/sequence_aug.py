import numpy as np 
import random 
from scipy.signal import resample


# ---- Preprocessing of the data ----

class Compose(object):
    "Lets us chain multiple transforms together and treat them as one transform."
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    "Reverse the order of dimentions. "
    "(B, C, L) -> (L, C, B)"
    def __call__(self, seq):
        return seq.transpose()  
    

class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)




# ----- Data Augmentation ---------

class AddGaussian(object):
    "Randomly add Gaussian noise into the input signal."
    def __init__(self, sigma = 0.01):
        self.sigma = sigma 
    def __call__(self, seq):
        # Since the centre of the normal distribution is around 0, scaling of the data should be done after the trandformation?
        return seq + np.random.normal(loc=0, scale = self.sigma, size = seq.shape)
    

class Scale(object):
    "Ranbdomly multiplies the input signal with a scaler, distributed (1, 0.01)"
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix
    

class RandomStretch(object):
    "Time-streach/time-warp data augmentetion."
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        seq_aug = np.zeros(seq.shape)
        T = seq.shape[1]
        length = int(T * (1 + (random.random()-0.5)*self.sigma))
        for i in range(seq.shape[0]):
            y = resample(seq[i, :], length)
            if length < T:
                if random.random() < 0.5:
                    seq_aug[i, :length] = y
                else:
                    seq_aug[i, T-length:] = y
            else:
                if random.random() < 0.5:
                    seq_aug[i, :] = y[:T]
                else:
                    seq_aug[i, :] = y[length-T:]
        return seq_aug


class FFT(object):
    """
    Two-sided FFT transform.
    Input:  (C, T)
    Output: (C, T)  (FFT magnitude, zero-frequency centered)
    """
    def __init__(self, normalize=False, eps=1e-8):
        """
        normalize: if True, divide by signal length
        """
        self.normalize = normalize
        self.eps = eps

    def __call__(self, seq):
        """
        seq: np.ndarray of shape (channels, time)
        """
        seq = np.asarray(seq)

        # FFT along time axis
        fft = np.fft.fft(seq, axis=-1)

        if self.normalize:
            fft = fft / (seq.shape[-1] + self.eps)

        # Shift zero-frequency component to center
        fft = np.fft.fftshift(fft, axes=-1)

        # Return magnitude (most ML models want real values)
        return np.abs(fft)



# This is the implementation from the project
def oneD_Fourier_transform(data: np.ndarray) -> np.ndarray:
    """
    Apply 1D FFT (magnitude) along the window dimension.
    data: (N,1,W)
    returns (N,1,W) magnitude (full spectrum length W for consistency)
    """
    N, C, W = data.shape
    assert C == 1
    out = np.abs(np.fft.fft(np.squeeze(data), axis=1))
    return out.reshape(N, 1, W)

    

class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq
        

# TODO: Add forth and back flipping (flips the signal up forth and back)
# TODO: ADD Tructation ( Tranctation masks a random part of the signal with aero values)
# TODO: Add Weighted moving average filter 


# ---- Scaling -------


class Normalize:
    def __init__(self, mode="zero_one", eps=1e-8):
        """
        mode:
          - "zero_one"  -> [0, 1]
          - "minus_one_one" -> [-1, 1]
          - "mean_std" -> zero mean, unit std
        """
        self.mode = mode
        self.eps = eps

    def __call__(self, seq):
        seq = seq.astype(float, copy=False)

        if self.mode == "zero_one":
            min_, max_ = seq.min(), seq.max()
            return (seq - min_) / (max_ - min_ + self.eps) # eps to avoid deviding by zero

        elif self.mode == "minus_one_one":
            min_, max_ = seq.min(), seq.max()
            return 2 * (seq - min_) / (max_ - min_ + self.eps) - 1

        elif self.mode == "mean_std":
            mean, std = seq.mean(), seq.std()
            return (seq - mean) / (std + self.eps)

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    "A composed transform pipeline for dataset preprocessing / augmentation depending on whether it's training or validation."
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            AddGaussian(),
            # TODO: Check why are there every one of them here, does it mean that they are all done allt he time?
            Scale(),
            RandomStretch(),
            RandomCrop(),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]
