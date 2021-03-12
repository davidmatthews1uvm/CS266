import taichi as ti
from time import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tqdm

ti.init(arch=ti.cpu)

amin = 2
amax = 4

burnin = 8192

NUM_A =  2048
sample_len = 900
da = (amax - amin)/NUM_A


xmin = -.2
xmax = 1.3
xrange = xmax - xmin

pixels = ti.field(ti.u8, (NUM_A, NUM_A))
points = ti.field(ti.f32, (NUM_A*sample_len, 2))
lyapunov_exp = ti.field(ti.f32, (NUM_A))

@ti.kernel
def logistic_iterate(burnin: ti.i32,
                    sample_len: ti.i32,
                    amin: ti.f64,
                    amax: ti.f64,
                    da: ti.f64,
                    n_threads: ti.i32):
    for tid in range(n_threads):
        x = ti.random(ti.f64)
        a = amin + tid * da
        for i in range(burnin):
            x = a * x * (1-x)
        
        summed_lyapunov_exp = 0.0

        for i in range(sample_len):
            summed_lyapunov_exp += ti.log(abs(a * (1 - 2 * x)))
            x = a * x * (1-x)
            px_row = int((x - xmin)/xrange * float(NUM_A))
            pixels[tid, px_row] = ti.cast(1, ti.u8)
            points[tid*sample_len + i, 0] = a 
            points[tid*sample_len + i, 1] = x 

        lyapunov_exp[tid] = summed_lyapunov_exp / sample_len

print("Running logistic map")
logistic_iterate(burnin, sample_len, amin, amax, da, NUM_A)
print("Done. Now plotting")
lyapunov_exp_np = lyapunov_exp.to_numpy()
pixels_np = pixels.to_numpy()
points_np = points.to_numpy()

points_np_sub = points_np[::20, :]
points_np_sub.shape


fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(points_np_sub[:, 0], points_np_sub[:, 1], s=0.1, alpha=0.3, label="Logistic Bifurcation Diagram")
ax.set_ylabel(f"Values of $x$", fontsize=18)
ax.set_xlabel(f"Parameter $a$", fontsize=18)
ax_twin = ax.twinx()
ax_twin.plot(np.arange(NUM_A)*da + amin, lyapunov_exp_np, c="C1", label="Lyapunov exponent")
ax_twin.axhline(0.0, c="C1", linestyle="dashed", alpha=0.5)
ax_twin.set_ylabel(f"Lyapunov exponent", color="C1", fontsize=18)
ax_twin.tick_params(axis='y', labelcolor="C1")

plt.savefig("a5_q5.pdf", bbox_inches="tight")
plt.savefig("a5_q5.png", dpi=NUM_A/5, bbox_inches="tight")
