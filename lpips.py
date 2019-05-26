# import sys; sys.path += ['models']
import torch
from util import util
from models import dist_model as dm
from IPython import embed
import os, sys
import numpy as np

use_gpu = True         # Whether to use GPU
spatial = False        # Return a spatial map of perceptual distance

## Initializing the model
model = dm.DistModel()

# Linearly calibrated models
model.initialize(model='net-lin',net='alex',use_gpu=use_gpu,spatial=spatial)

# Low-level metrics
print('Model [%s] initialized'%model.name())

files = os.listdir(sys.argv[1])
index = np.load(sys.argv[2])
lpips = np.zeros(10)
for i in range(10):
	p = index[i]
	score = 0.
	for j in range(1900):
		img1 = util.im2tensor(util.load_image(files[j]))
		img2 = util.im2tensor(util.load_image(files[p[j]]))
		score += model.forward(ex_ref,ex_p0)
	lpips[i] = score / 1900.

print(mp.mean(lpips), np.std(lpips))