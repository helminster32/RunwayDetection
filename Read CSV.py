# Import pandas library
import pandas as pd
import os
import numpy
import re
import numpy as np
import cv2

root_folder = '.\DataBase\Labels'

# Inicializo dataframe
data = {'Full path': None, 'Coordinates': None}
df = pd.DataFrame(data, columns = ['Full path', 'Coordinates'])
#df = df.append({'Full path': '/hola/la.jpg', 'Coordinates': [34, 45, 44, 44]}, ignore_index=True)
#print(df.shape)
#print(df.head(5))

# Escanear una carpeta (y subcarpetas) para buscar csv.
csv_list = []

for root, dirs, files in os.walk(root_folder, topdown=False):
   for name in files:
      filename, file_extension = os.path.splitext(name)
      if file_extension == '.csv':
          csv_list.append(os.path.join(root, name))

print(csv_list)

# Procesar cada csv
for i in csv_list:
    with open(i) as fp:
        for linea in enumerate(fp):
            a = str(linea)
            xstring = re.search(r'(?<=x"":)\[[0-9]{1,3},[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}]', a).group()
            X = re.findall(r"[0-9]{1,3}", xstring)
            ystring = re.search(r'(?<=y"":)\[[0-9]{1,3},[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}]', a).group()
            Y = re.findall(r"[0-9]{1,3}", ystring)
            x1 = X[0]
            x2 = X[1]
            x3 = X[2]
            x4 = X[3]

            y1 = Y[0]
            y2 = Y[1]
            y3 = Y[2]
            y4 = Y[3]

            imagename = i.replace(".csv","",1).replace("\Labels","",1) + "\\" + re.search(r'[0-9]*.jpg', a).group()

            df = df.append({'Full path': imagename, 'Coordinates': [x1, x2, x3, x4, y1, y2, y3, y4]}, ignore_index=True)

df.to_csv('./DataBase/Total/gt.csv')
