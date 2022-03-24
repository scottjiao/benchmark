import os

import numpy as np


for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if ".w.npy" in name:
            fn=os.path.join(root, name)
            temp=np.load(fn,allow_pickle=True)
            print(temp)