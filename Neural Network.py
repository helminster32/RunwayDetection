import re
import numpy as np
from PIL import Image
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import keras.backend as K


filepath = "DataBase\\via_region_data.csv"
coordinates = []
images = []

with open(filepath) as fp:
   for linea in enumerate(fp):
       a = str(linea)
       x1 = int(re.search(r'(?<=x"":\[)[0-9]*', a).group())
       x2 = int(re.search(r'(?<=x"":\[[0-9]{3},)[0-9]{3}', a).group())
       x3 = int(re.search(r'(?<=x"":\[[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())
       x4 = int(re.search(r'(?<=x"":\[[0-9]{3},[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())

       y1 = int(re.search(r'(?<=y"":\[)[0-9]*', a).group())
       y2 = int(re.search(r'(?<=y"":\[[0-9]{3},)[0-9]{3}', a).group())
       y3 = int(re.search(r'(?<=y"":\[[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())
       y4 = int(re.search(r'(?<=y"":\[[0-9]{3},[0-9]{3},[0-9]{3},)[0-9]{3}', a).group())

       coordinates.append([x1,x2,x3,x4,y1,y2,y3,y4])

       imagename = ".\\DataBase\\" + re.search(r'[0-9]*.jpg', a).group()
       CurrentImage = Image.open(imagename)
       images.append(np.asarray(CurrentImage))

x = np.asarray(images, dtype=np.float32)
print(x)
y = np.asarray(coordinates, dtype=np.float32)
print(y)
