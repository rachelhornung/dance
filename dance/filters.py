'''
Created on Oct 13, 2016

@author: horn_ra (Rachel Hornung)

Module to preprocess emg signal.

'''

import numpy as np
import copy
from scipy.fftpack import dct

'''
most of the functions are implemented according to
Angkoon Phinyomark, Pronchai Phukpattaranont, Chusak Limaskul (2012): 
"Feature selection for EMG signal classification" 
Expert Systems with Applications 39, pp. 7420-7431
mfcc, filter_bank, and power spectrum based on 
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fn:1 
'''

EPS = 1e-16


def normalize(x):
    import sklearn.preprocessing as sk_prep
    """
    Scale observations of the emg environment to have 0 mean and unit variance. Each channel is normalized independently.
    """
    preprocessor = sk_prep.StandardScaler(with_std=True)

    preprocessor.fit(x)
    return preprocessor.transform(x)


def fft(sig, apply_hamming=True, band_pass_filter=None, sampling_rate=None):
    """
    :param sig: signal with shape n_samples, n_channels, time
    :param apply_hamming: if True the hamming window is applied to each window before computing FFT.
            See :func:``numpy.hamming`` for more details.
    :param band_pass_filter:
    :param sampling_rate:
    :return:
    """

    nfft = sig.shape[-1]
    if apply_hamming:
        hamming_window = np.hamming(nfft)
        sig *= hamming_window
    amp = abs(np.fft.rfft(sig, axis=-1))
    # TODO: calculate separate freq for different sampling rates
    freq = np.fft.rfftfreq(nfft, d=1. / sampling_rate[0])
    if band_pass_filter:
        if sampling_rate is None:
            raise ValueError(
                'When performing a band pass filter, sampling rate must be specified')
        min_freq, max_freq = band_pass_filter
        valid_freq, = np.where(
            np.logical_and.reduce((min_freq <= freq, freq <= max_freq)))
        amp = amp[:, valid_freq, :]
        freq = freq[valid_freq]
    return amp, freq


def power_spectrum(sig, apply_hamming=True, band_pass_filter=None, sampling_rate=None):
    amp, freq = fft(sig, apply_hamming, band_pass_filter, sampling_rate)
    nfft = sig.shape[-1]
    pow = (1. / nfft) * (amp**2)
    return pow


def filter_bank(sig, apply_hamming=True, band_pass_filter=None, sampling_rate=None):
    pow = power_spectrum(sig, apply_hamming, band_pass_filter, sampling_rate)
    nfft = sig.shape[-1]
    nfilt = 40
    low_freq_mel = 0
    # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (sampling_rate[0] / 2) / 700))
    # Equally spaced in Mel scale
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / sampling_rate[0])

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow, fbank.T)
    # Numerical Stability
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks


def mfcc(sig, apply_hamming=True, band_pass_filter=None, sampling_rate=None, cep_lifter=None):
    filter_banks = filter_bank(
        sig, apply_hamming, band_pass_filter, sampling_rate)
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[
        :, :, 1: (num_ceps + 1)]  # Keep 2-13
    if cep_lifter is not None:
        (nsamples, nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
    return mfcc


def bandstop(sig, at_freq, sampling_rate, filter_out_all_harmonics=False):
    """
    Filter-out given frequency from the signal.
    :param at_freq frequency to filter
    :param filter_out_all_harmonics if True, will filter out all multiples of the frequency as well.
    """
    import scipy.signal
    sig = np.copy(sig.obs)
    nyquist_freq = 0.5 * sampling_rate
    for i in range(int(nyquist_freq) // at_freq if filter_out_all_harmonics else 1):
        fr = at_freq * (i + 1)
        w = fr / nyquist_freq
        b, a = scipy.signal.iirnotch(w, 1)
        print(b, a)
        sig = scipy.signal.filtfilt(b, a, sig, axis=0)
    return sig


def rescale(x, min=0, max=1):
    """
    Rescale the signal to be in [min, max]
    """
    n_features = x.shape[-1]
    x_copy = np.copy(x)
    for f in range(n_features):
        c = x.obs[:, :, f]
        c_min = np.min(c)
        c_max = np.max(c)
        x_copy[:, :, f] = (c - c_min) / (c_max - c_min) * (max - min) + min
    return x_copy


def clip(x, min=0, max=1):
    """
    Clip signal to be in [min, max]
    """
    return np.clip(x.obs, min, max)


def zero_crossings(emg, threshold):
    # calculate the number of zero crossing of the signal in each window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[2] > 1:
        zc_values = np.sign(-emg[:, :, :-1] * emg[:, :, 1:]) * (
            abs(emg[:, :, :-1] - emg[:, :, 1:]) > threshold[np.newaxis, :, np.newaxis])
        zc_changes = np.zeros_like(zc_values)
        zc_changes[zc_values > 0] = 1
        sum_sign_diff = np.sum(zc_changes, axis=2)
    else:
        sum_sign_diff = np.zeros((emg.shape[0], emg.shape[1]))
    return sum_sign_diff


def slope_sign_changes(emg, threshold):
    # calculate the number of slope sign changes of the signal in each window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[2] > 2:
        diff = emg[:, :, :-1] - emg[:, :, 1:]
        ss_values = (-diff[:, :, :-1]) * diff[:, :, 1:]
        ss_changes = np.zeros_like(ss_values)
        ss_changes[ss_values > threshold[np.newaxis, :, np.newaxis]] = 1
        sum_sign_diff_diff = np.sum(ss_changes, axis=2)
    else:
        sum_sign_diff_diff = np.zeros((emg.shape[0], emg.shape[1]))
    return sum_sign_diff_diff


def wave_length(emg):
    # calculate the wave length of the signal for each window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[2] > 1:
        diff = emg[:, :, 1:] - emg[:, :, :-1]
        wl = np.sum(abs(diff), axis=2)
    else:
        wl = np.zeros((emg.shape[0], emg.shape[1]))
    return wl


def average_amplitude_change(emg):
    # calculate the average wave length within a window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    wl = wave_length(emg)
    wl /= n_time_steps
    return wl


def difference_abs_std_deviation_value(emg):
    # calculate the std of the absolute differences
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    if emg.shape[2] > 1:
        diff = emg[:, :, 1:] - emg[:, :, :-1]
        dasdv = np.sum(diff**2, axis=2)
        dasdv /= (n_time_steps - 1)
    else:
        dasdv = np.zeros((emg.shape[0], emg.shape[1]))
    return dasdv


def mean_abs(emg):
    # calculate the mean of the absolute value of the signal for each window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[2] > 1:
        ab = abs(emg)
        ma = np.mean(abs(ab), axis=2)
    else:
        ma = abs(emg)
    return ma


def mean_abs_value_slope(emg):
    # calculate the mean average value slope for each windpw
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[0] > 1:
        mav = mean_abs(emg)
        mavs = mav[1:, :] - mav[:-1, :]
    else:
        mavs = np.zeros(1, emg.shape[1])
    return mavs


def integral(emg):
    # calculate the square integral over windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    if emg.shape[2] > 1:
        sq = emg**2
        inte = np.sum(sq, axis=2)
    else:
        inte = emg**2
    return inte


def variance(emg):
    # calculate the variance of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    inte = integral(emg)
    if n_time_steps > 1:
        vari = inte / (n_time_steps - 1)
    else:
        vari = np.zeros_like(inte)
    return vari


def third_temp_moment(emg):
    # calculate the third temporal moment of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    if n_time_steps > 1:
        tm = emg**3
        tm = np.sum(abs(tm), axis=2)
        tm /= n_time_steps
    else:
        tm = abs(emg**3)
    return tm


def fourth_temp_moment(emg):
    # calculate the fourth temporal moment of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    if n_time_steps > 1:
        fm = emg**4
        fm = np.sum(fm, axis=2)
        fm /= n_time_steps
    else:
        fm = emg**4
    return fm


def fifth_temp_moment(emg):
    # calculate the fifth temporal moment of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    if n_time_steps > 1:
        fm = emg**5
        fm = np.sum(abs(fm), axis=2)
        fm /= n_time_steps
    else:
        fm = abs(emg**5)
    return fm


def rms(emg):
    # calculate the root mean square value of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    if emg.shape[2] > 1:
        rm = emg**2
        rm = np.sum(rm, axis=2)
        rm /= n_time_steps
        rm = np.sqrt(rm)
    else:
        rm = abs(emg)
    return rm


def moving_average(emg, window_size=50):
    smoothed_emg = np.zeros_like(emg)
    for i in range(emg.shape[0]):
        offset = min(i + 1, window_size)
        smoothed_emg[i] = np.sum(emg[i - offset + 1:i + 1], axis=0) / offset
    return smoothed_emg


def moving_max(emg, window_size=50):
    from scipy.ndimage.filters import maximum_filter1d
    return maximum_filter1d(emg, size=window_size, axis=0)


def log_detector(emg):
    # calculate the log power of the windowed emg
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    n_time_steps = emg.shape[2]
    l = np.log(abs(emg) + EPS)
    if n_time_steps > 1:
        l = np.sum(l, axis=2)
        l /= n_time_steps
    l = np.exp(l)
    return l


def myopulse_percentage_rate(emg, threshold):
    # calculate the frequency of surpassing the thresholdvalue within
    # each emg window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    emg = copy.deepcopy(emg)

    def f(x):
        x[abs(x) < threshold[np.newaxis, :, np.newaxis]] = 0
        x[abs(x) >= threshold[np.newaxis, :, np.newaxis]] = 1
        return x

    n_time_steps = emg.shape[2]
    if n_time_steps > 1:
        mpr = np.sum(f(emg), axis=2)
    else:
        mpr = f(emg)
    return mpr


def willison_amplitude(emg, threshold):
    # calculate the frequency of surpassing the threshold
    # when observing the slope length between to adjacent
    # time steps in an emg window
    # emg of shape [n_samples, n_sensors, n_time_steps]
    ## returns in shape [n_samples, n_sensors]
    def f(x):
        x[abs(x) < threshold[np.newaxis, :, np.newaxis]] = 0
        x[abs(x) >= threshold[np.newaxis, :, np.newaxis]] = 1
        return x

    n_time_steps = emg.shape[2]
    if n_time_steps > 1:
        diff = emg[:, :, :-1] - emg[:, :, 1:]
        wamp = np.sum(f(diff), axis=2)
    else:
        wamp = np.zeros((emg.shape[0], emg.shape[1]))
    return wamp


def windowify_sequence(seq, window_size, window_step):
    """
    Given a sequence, scans it from the beginning and for every window_step-th element creates a 'window' of window_size
    elements, starting with the current one.
    :param seq - a tensor of shape (num_samples, sequence_length)
    :returns a tensor of shape (num_samples, sequence_length / window_step, window_size)
    """
    windowed_seq = []
    # print(seq.shape)
    for o in range(0, seq.shape[1] - window_size + 1, window_step):
        windowed_seq.append(
            seq[:, o: o + window_size]
        )
    windowed_seq = np.array(windowed_seq)
    return np.rollaxis(windowed_seq, 1)
