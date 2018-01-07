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
no check whether label has been verified
songs with multiple assigned labels are added to data of both labels
if "getting labeled dance music.ipynb" is rerun, old labels are lost
according to https://mediatum.ub.tum.de/doc/1138535/582915.pdf# classification rate should be very good -> more suitable audio features? use the ones from the paper
no hyperparameter search



********************************
models in use:
RandomForestClassifier (as baseline)
NN

models to be implemented
RNN
LSTM
CNN

********************************
audio preprocessing in use:
raw
rectified
"emg" features
mfcc & filter banks

models to be implemented
audio features like beats per minute (https://mediatum.ub.tum.de/doc/1138535/582915.pdf#)


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

