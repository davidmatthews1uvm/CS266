import taichi as ti
from time import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tqdm

ti.init(arch=ti.cpu)

amin = 1.935
amax = 1.952
# amin = 1.964
# amax = 1.967
b = -0.3
burnin = int(1e6)

NUM_A = 1<<13 # 1<<15
sample_len = NUM_A<<2

da = (amax - amin)/NUM_A

IC = (0.0, 2.0)

xmin = -0.75
xmax = 2.25
xrange = xmax - xmin

pixels = ti.field(ti.f32, (NUM_A, NUM_A))
x_field = ti.field(ti.f32, (NUM_A, NUM_A))

@ti.kernel
def henon_iterate(burnin: ti.i32,
                    sample_len: ti.i32,
                    amin: ti.f32,
                    amax: ti.f32,
                    da: ti.f32,
                    n_threads: ti.i32):
    for tid in range(n_threads):
        x = IC[0]
        y = IC[1]
        a = amin + tid * da
        for i in range(burnin):
            x_tmp = a - x*x + b * y
            y = x
            x = x_tmp
        for i in range(sample_len):
            x_tmp = a - x*x + b * y
            y = x
            x = x_tmp
            x_field[tid, i] = x
            px_row = int((x - xmin)/xrange * float(NUM_A))
            pixels[tid, px_row] = 1.0


henon_iterate(burnin, sample_len, amin, amax, da, NUM_A)
pixels_np = pixels.to_numpy()

print("now plotting...")
SUBSAMPLE = 2

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(pixels_np[::SUBSAMPLE, ::-1*SUBSAMPLE].T, cmap="gray", interpolation="nearest")

# locs, labels = plt.xticks()
locs = range(0, NUM_A//SUBSAMPLE, (NUM_A//SUBSAMPLE)//40)
plt.xticks(locs, ["{:.5f}".format(amin + da *loc*SUBSAMPLE) for loc in locs], rotation=90)

locs, labels = plt.yticks()
plt.yticks(locs, ["{:.6f}".format(((pixels_np.shape[1] - loc *  SUBSAMPLE)* xrange)/NUM_A + xmin) for loc in locs])
plt.savefig("q5_labeled.png", dpi=NUM_A/5, bbox_inches="tight")
# plt.show()



# im = Image.fromarray(pixels_np[:, ::-1].T.astype('uint8')*255)
# im.save("q5_raw.png")
# # im.show()