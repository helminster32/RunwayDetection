import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

imagename = ".\\DataBase\\06100.jpg"
CurrentImage = Image.open(imagename)
CurrentImage.show()
print(CurrentImage.size)
print(CurrentImage.mode)
print(CurrentImage.format)

# Para convertir la imagen a escala de grises
#CurrentImageGris = CurrentImage.convert('L')
#CurrentImageGris.show()


