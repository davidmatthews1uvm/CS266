import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle

sol = pickle.load(open("a12q1_sln.pkl", "rb"))

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(sol.y[0,:10000], sol.y[1,:10000], sol.y[2,:10000], 'blue')
plt.savefig("a12q1b.pdf", bbox_inches="tight")
plt.show()
