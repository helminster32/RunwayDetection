from PIL import Image
import re

filepath = "DataBase\\via_region_data.csv"

with open(filepath) as fp:
   for linea in enumerate(fp):
       a = str(linea)
       imagename = ".\\DataBase\\" + re.search(r'[0-9]*.jpg', a).group()
       imagedestiny = ".\\DataBase Reescalada\\" + re.search(r'[0-9]*.jpg', a).group()
       img = Image.open(imagename)
       img = img.resize((320, 240), Image.ANTIALIAS)
       img.save(imagedestiny)
