import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle

TIMESPAN = 1_000_000
NUM_STEPS_PER_TIME_UNIT = 100
def func(_, curr_state, sigma=10, b = 8/3, r=28):

    x, y, z = curr_state

    # compute the partial derivatives
    fx = sigma * (y - x)
    fy = r * x - y - x * z
    fz = x * y - b * z

    return np.array([fx, fy, fz], float)

r0 = [-12.6480, -13.9758, 30.9758]
sol = solve_ivp(func, [0, TIMESPAN], r0, t_eval=np.linspace(0, TIMESPAN, NUM_STEPS_PER_TIME_UNIT * TIMESPAN ), vectorized=True)

# and plot it
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(sol.y[0,:], sol.y[1,:], sol.y[2,:], 'blue')
plt.show()

pickle.dump(sol, open("a12q1_sln.pkl", "wb"))

# Normalize each axis.
sol_normalized = sol.y.copy() 
sol_normalized[:, :] -= sol_normalized.min(axis=1).reshape(3, -1)
sol_normalized[:, :] /= (sol_normalized.max(axis=1).reshape(3, -1) + 1e-6)

cells_filled = []
for i in range(10):
    n_cells = 1<<i
    grid = np.zeros((n_cells, n_cells, n_cells))
    flat_grid = grid.flatten()
   
    grid_cells_visited = (sol_normalized * n_cells).astype(int)
   
    flat_grid[grid_cells_visited[0, :] + n_cells * grid_cells_visited[1, :] + grid_cells_visited[2, :] * n_cells**2] = 1
    grid = flat_grid.reshape(grid.shape)
    cells_filled.append(( n_cells, grid.sum()))


cells_filled_log = np.log(np.array(cells_filled).T)
slope, intercept, r, p, se = linregress(cells_filled_log[0], cells_filled_log[1])
print(slope)

plt.scatter(cells_filled_log[0], cells_filled_log[1])
plt.plot([0, cells_filled_log[0].max()],[intercept, slope*cells_filled_log[0].max() + intercept])
plt.title("Measured Fractal Dimension: {:.4f}".format(slope))
plt.ylabel("$\ln(n_{cells\_filled})$")
plt.xlabel("$\ln(n_{cells})$")
plt.savefig("a12q1.pdf")
plt.show()