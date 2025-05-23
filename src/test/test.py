import pprint

import numpy as np
from PIL import Image

a = Image.open("preds/SC-STO-250422/expo50_gain60/CLAHE/250425_900_fil-6_O2-0/0/0.0.png")
a = np.array(a)


pprint.pprint(a)
