import re
import numpy as np
import cv2
import nets
from keras.applications.resnet50 import ResNet50

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
       CurrentImage = cv2.imread(imagename)
       images.append(np.asarray(CurrentImage))

x = np.asarray(images, dtype=np.float32)
print(x.shape)
y = np.asarray(coordinates, dtype=np.float32)
print(y.shape)

# Input
img_input = Input(shape=(240, 320, 3))

# ResNet50
model = ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
x = model.output

# FC layers
x = Flatten()(x)
x = Dense(1024)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(output_dim)(x)
x = Activation('softmax')(x)

# Define steering-collision model
model = Model(inputs=[img_input], outputs=[x])











