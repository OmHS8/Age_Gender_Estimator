from PIL import Image
import numpy as np

img_path = 'src\pipeline\IMG20220727195325.jpg'
img = Image.open(img_path).convert('L')
img.show()
pixels = np.asarray(img)
print(pixels.shape)