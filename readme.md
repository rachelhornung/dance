little side project: code to learn ballroom dance style from audio
goals:
-web page scraping
-data gathering from the web
-base line check
-getting aquainted with keras
-evaluate different ml/deep models and audio preprocessing techniques

training data downloaded from youtube
labels from www.tanzmusik-online.de


********************************
current caveeats:
songs with multiple assigned labels are added to data of both labels;
-> should have an implementation that displays likelihood of different labels and compares to all possible labels

according to https://mediatum.ub.tum.de/doc/1138535/582915.pdf# classification rate should be very good -> more suitable audio features? use the ones from the paper;

no hyperparameter search

training is memory limited -> use batch wise data generator from keras



********************************
models in use:
RandomForestClassifier (as baseline)
NN
RNN
LSTM
CNN

models to be implemented
multi-label SVM
other models with multi-label prediction -> requires other means of assessment

********************************
audio preprocessing in use:
raw
rectified
"emg" features
mfcc & filter banks

models to be implemented
audio features like beats per minute (https://mediatum.ub.tum.de/doc/1138535/582915.pdf#)

********************************


********************************

depends on:
bs4.BeautifulSoup
keras
numpy 
os.path
pandas
scipy.io
scipy.fftpack
sklearn
tensorflow
time
urllib2

