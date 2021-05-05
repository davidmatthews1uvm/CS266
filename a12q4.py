import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
import pickle

from tqdm import tqdm 

TIMESPAN = 2_00
NUM_STEPS_PER_TIME_UNIT = 100
def func(_, curr_state, sigma=10, b = 8/3, r=28):

    x, y, z = curr_state

    # compute the partial derivatives
    fx = sigma * (y - x)
    fy = r * x - y - x * z
    fz = x * y - b * z

    return np.array([fx, fy, fz], float)

r0 = [-12.6480, -13.9758, 30.9758]
x_coords = []
y_coords = []
for r in tqdm(range(345)):
    func_r = partial(func, r=r)
    sol = solve_ivp(func_r, [0, TIMESPAN], r0, t_eval=np.linspace(0, TIMESPAN, NUM_STEPS_PER_TIME_UNIT * TIMESPAN ), vectorized=True)
    z =  sol.y[2, 100:]
    z_maxes = z[np.r_[True, z[1:] > z[:-1]] & np.r_[z[:-1] > z[1:], True]]
    x_coords += [r]*len(z_maxes)
    y_coords += list(z_maxes)

plt.scatter(x_coords, y_coords, s=0.01)
plt.savefig("a12q4b.pdf", bbox_inches="tight")
plt.savefig("a12q4b.png", bbox_inches="tight", dpi=300)
plt.show()



sol = solve_ivp(func, [0, 1000], r0, t_eval=np.linspace(0, 1000, NUM_STEPS_PER_TIME_UNIT * 1000 ), vectorized=True)
z = sol.y[2]
z_maxes = z[np.r_[True, z[1:] > z[:-1]] & np.r_[z[:-1] > z[1:], True]]
bounds = [28,  50]

plt.scatter(z_maxes[:-1], z_maxes[1:], s=0.1)
plt.plot(bounds, bounds, c="k", linestyle="dashed")
plt.savefig("a12q4a.pdf", bbox_inches="tight")
plt.savefig("a12q4a.png", bbox_inches="tight", dpi=300)
plt.show()
