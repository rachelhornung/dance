import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils

import dance.filters as f
from dance.sampling import subsample_mean as subsample


def get_selected_dances_only(data, selected_dances):
    return data[data["label"].isin(selected_dances)]


def get_samples(data):
    return np.stack(data["sample"].as_matrix())


def get_rate(data):
    return data["rate"].as_matrix()


def get_labels(data):
    return data["label"].as_matrix()


class BaseAudioSamples(object):

    def __init__(self, data, subsample_rate=1,
                 selected_dances=["Cha Cha Cha", "Disco Fox",
                                  "Jive", "Langsamer Walzer",
                                  "Rumba", "Salsa", "Tango",
                                  "Wiener Walzer"]):
        data_ = get_selected_dances_only(data, selected_dances)
        self.selected_dances = selected_dances
        self.subsample_rate = subsample_rate
        self.original_rate = get_rate(data_)
        self.rate = self.original_rate / self.subsample_rate
        self.n_dances = len(selected_dances)
        self.X = subsample(get_samples(data_).T, subsample_rate).T
        self.Y = get_labels(data_)
        self.stratified = False
        self.normed = False
        self.one_hot = False

    def count_samples(self):
        dance_counts = {}
        for dance in self.selected_dances:
            dance_counts[dance] = sum(self.Y == dance)
        return dance_counts

    def norm(self, scaler=None):
        if not self.normed:
            if scaler is None:
                self.scaler = StandardScaler()
            else:
                self.scaler = scaler
            self.X = self.scaler.fit_transform(self.X)
            self.normed = True
        return self.scaler

    def encode(self, encoders=None):
        if not self.one_hot:
            if encoders is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(self.Y)
                # self.onehot_encoder = OneHotEncoder()
                # self.onehot_encoder.fit(
                # np.reshape(self.label_encoder.transform(self.Y), (-1, 1)))
            else:
                self.label_encoder = encoders
                # self.onehot_encoder = encoders[1]
            self.Y = np_utils.to_categorical(
                self.label_encoder.transform(self.Y))
        return self.label_encoder

    def get_scaler(self):
        return self.scaler

    def stratify(self):
        if not self.stratified:
            # get same number of samples for each dance
            dance_counts = self.count_samples()
            # the dance that has the fewest samples is used as reference
            least_dance = min(dance_counts, key=dance_counts.get)
            desired_samples = dance_counts[least_dance]
            # print("number desired samples", desired_samples)
            for dance in self.selected_dances:
                if dance != least_dance:
                    # get percentage of data to keep of respective dance
                    # add slight margin so that there will hardly less samples
                    # than desired_samples
                    keep_rate = (
                        1 - (desired_samples / dance_counts[dance])) * 0.98
                    # get random array indicating whether an element should be kept
                    # decided upon keep_rate
                    keeps = np.random.rand(self.X.shape[0])
                    keeps[keeps > keep_rate] = 1
                    keeps[keeps <= keep_rate] = 0
                    # either an element is not of type dance or
                    # keeps decides whether the sample should stay
                    self.X = self.X[np.any([self.Y != dance, keeps], axis=0)]
                    self.Y = self.Y[np.any([self.Y != dance, keeps], axis=0)]
            self.stratified = True

    def get_as_timeseries(self):
        return np.reshape(self.X, (self.X.shape[0], -1, 1))


class RectifiedAudioSamples(BaseAudioSamples):

    def __init__(self, data, subsample_rate=1,
                 selected_dances=["Cha Cha Cha", "Disco Fox",
                                  "Jive", "Langsamer Walzer",
                                  "Rumba", "Salsa", "Tango",
                                  "Wiener Walzer"]):
        super(RectifiedAudioSamples, self).__init__(
            data, subsample_rate, selected_dances)
        self.X = abs(self.X)


class FeaturedAudioSamples(BaseAudioSamples):

    def __init__(self, data, subsample_rate=1,
                 selected_dances=["Cha Cha Cha", "Disco Fox",
                                  "Jive", "Langsamer Walzer",
                                  "Rumba", "Salsa", "Tango",
                                  "Wiener Walzer"],
                 threshold=100):
        super(FeaturedAudioSamples, self).__init__(
            data, subsample_rate, selected_dances)
        self.threshold = threshold
        self.X = self.calc_features()

    def calc_features(self):
        self.X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[-1]))
        aac = f.average_amplitude_change(self.X)
        dasdv = f.difference_abs_std_deviation_value(self.X)
        fitm = f.fifth_temp_moment(self.X)
        fotm = f.fourth_temp_moment(self.X)
        i = f.integral(self.X)
        ld = f.log_detector(self.X)
        ma = f.mean_abs(self.X)
        # mavs = features.mean_abs_value_slope(self.X)
        mpr = f.myopulse_percentage_rate(
            self.X, np.ones((self.X.shape[1],)) * self.threshold)
        rms = f.rms(self.X)
        ssc = f.slope_sign_changes(
            self.X, np.ones((self.X.shape[1],)) * self.threshold)
        ttm = f.third_temp_moment(self.X)
        var = f.variance(self.X)
        wl = f.wave_length(self.X)
        wa = f.willison_amplitude(
            self.X, np.ones((self.X.shape[1],)) * self.threshold)
        zc = f.zero_crossings(
            self.X, np.ones((self.X.shape[1],)) * self.threshold * 20)
        X = np.hstack((aac, dasdv, fitm, fotm, i, ld,
                       ma, mpr, rms, ssc, ttm, var,
                       wl, wa, zc))

        return X


class MfccAudioSamples(BaseAudioSamples):

    def __init__(self, data, subsample_rate=1,
                 selected_dances=["Cha Cha Cha", "Disco Fox",
                                  "Jive", "Langsamer Walzer",
                                  "Rumba", "Salsa", "Tango",
                                  "Wiener Walzer"],
                 window_size=512,
                 filter_bank=False,
                 pre_emphasis=0.97,
                 cep_lifter=None):
        super(MfccAudioSamples, self).__init__(
            data, subsample_rate, selected_dances)
        self.window_size = window_size
        self.pre_emphasis = pre_emphasis
        self.filter_bank = filter_bank
        self.cep_lifter = cep_lifter
        self.X = self.calc_features()

    def calc_features(self):
        # pre emphasize signal to
        # (1) balance the frequency spectrum (high f usually have smaller magnitudes)
        # (2) avoid numerical problems during FFT
        # (3) improve the SNR.
        emph_X = np.hstack(
            (self.X[:, :1], self.X[:, 1:] - self.pre_emphasis * self.X[:, :-1]))
        if self.window_size == -1:
            emph_X = np.reshape(emph_X, (emph_X.shape[0], 1, emph_X.shape[-1]))
        else:
            emph_X = f.windowify_sequence(
                emph_X, self.window_size, int(self.window_size / 2))

        # amp, feq = f.fft(
        #     emph_X, apply_hamming=True, sampling_rate=self.rate)
        # pow = f.power_spectrum(
        #     emph_X, apply_hamming=True, sampling_rate=self.rate)
        if self.filter_bank:
            coefficients = f.filter_bank(
                emph_X, apply_hamming=True, sampling_rate=self.rate)
        else:
            coefficients = f.mfcc(
                emph_X, apply_hamming=True, sampling_rate=self.rate, cep_lifter=self.cep_lifter)
        coefficients -= (np.mean(coefficients, axis=1, keepdims=True) + 1e-8)
        return np.reshape(coefficients, (coefficients.shape[0], -1))

# TODO: implement beat and timing detection according to
# https://mediatum.ub.tum.de/doc/1138535/582915.pdf#
