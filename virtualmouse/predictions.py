%load_ext autoreload
%autoreload 2

from virtualmouse.vidcapture import capture
from virtualmouse.constants import ModelConsts
from virtualmouse.utils import LoadModel

from keras import Sequential
model = LoadModel()

def predCircle():
    return circle = [(rw,rh), 5, (255,123,123), 5]
