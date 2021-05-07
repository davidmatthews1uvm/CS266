import numpy as np
import matplotlib.pyplot as plt

def jacobian(x,y,z, r=28., omega=10., b= 8./3.):
    return np.array([[-omega, omega, 0],
                    [r - z, -1, -x],
                    [y, x, -b]])

fig, axs = plt.subplots(3,3, figsize=(9,9))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
N = 10_000
for n, z in enumerate([15, 25, 35]):
    for m, r in enumerate([12, 24.5, 28]):
        pos = np.zeros((N, 3))
        pos[:, :2] = np.random.random((N, 2))*50 - 25
        for idx in range(N):
            J = jacobian(pos[idx, 0], pos[idx, 1], z, r=r)
            max_e = np.linalg.eig(J)[0].max()
            pos[idx, 2] = max_e
        im = axs[n,m].scatter(pos[:, 0], pos[:,1], c=pos[:, 2], s=1, cmap="rainbow")
        axs[n,m].set_title("$R: {:.1f}\; |\; Z: {:d}$".format(r, z))
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
fig.colorbar(im, cax=cbar_ax, cmap="rainbow")
plt.savefig("a12q5.pdf", bbox_inches="tight")
