import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' or 'MacOSX'

import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("data/human_annotated_images/25.png")

plt.imshow(img)
plt.axis("off")
plt.title("Test Image Display")
plt.show()
