import scipy.signal as signal
import mne
import numpy as np


def bandpass_butt(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass]) / (fs / 2)
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 2), (hpass + 10)]) / (fs / 2)
    # % 通带允许最大衰减为 db
    alpha_pass = 1

    #  % 阻带允许最小衰减为  db
    alpha_stop = 5
    # % 获取阶数和截止频率
    N, Wn = signal.buttord(Wp, Ws, alpha_pass, alpha_stop)

    b, a = signal.butter(N, Wn, 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new


def bandpass_cheby1(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass]) / (fs / 2)
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 2), (hpass + 5)]) / (fs / 2)
    # % 通带允许最大衰减为 db
    alpha_pass = 3

    #  % 阻带允许最小衰减为  db
    alpha_stop = 20
    # % 获取阶数和截止频率
    N, Wn = signal.cheb1ord(Wp, Ws, alpha_pass, alpha_stop)

    b, a = signal.cheby1(N, 0.5, Wn, 'bandpass')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

def bandstop_cheby1(sig, lpass, hpass, fs, axis=-1):
    # wn1=2*freq0/srate
    # wn2=2*freq1/srate
    # % 通带的截止频率为2.75 hz - -75hz, 有纹波
    Wp = np.array([lpass, hpass]) / (fs / 2)
    #  % 阻带的截止频率
    Ws = np.array([(lpass - 2), (hpass + 5)]) / (fs / 2)
    # % 通带允许最大衰减为 db
    alpha_pass = 3

    #  % 阻带允许最小衰减为  db
    alpha_stop = 20
    # % 获取阶数和截止频率
    N, Wn = signal.cheb1ord(Wp, Ws, alpha_pass, alpha_stop)

    b, a = signal.cheby1(N, 0.5, Wn, 'bandstop')
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

def peak(sig, tar, fs, axis=-1):
    
    Q = 2.0  # Quality factor
    # Design peak filter
    b, a = signal.iirpeak(tar, Q, fs)
    sig_new = signal.filtfilt(b, a, sig, axis=axis)
    return sig_new

def resample(sig, orginFreq, newFreq, axis=-1):
    return mne.filter.resample(sig, down=orginFreq, up=newFreq, npad='auto', axis=axis)




######### tools to compute task component ############

def sig_var(data):
    n = data.shape[0]
    n_t = data.shape[1]

    var_sig_m = 0
    temp_cov = 0

    for i in range(n):
        for j in range(n):
            cov = np.cov(data[i, :], data[j, :])
            var_sig_m += cov[0][1]
            if i != j:
                temp_cov += cov[0][1]
    var_sig_m /= n * n
    var_sig = var_sig_m * n - temp_cov / n
    return var_sig, var_sig_m


def task_comp(data):
    n = data.shape[0]
    n_t = data.shape[1]

    x_m = np.mean(data, axis=0)
    var, var_m = sig_var(data)

    rho = np.sqrt(var / var_m)
    # task_sig=x_m*rho
    task_sig = np.zeros(n_t)

    for i in range(n_t):
        # task_sig[i]=rho*x_m[i]-(rho-1)*mu[i]
        # task_sig[i] = rho * (x_m[i]-mu[i])+mu[i]
        task_sig[i] = rho * (x_m[i])
    return task_sig

