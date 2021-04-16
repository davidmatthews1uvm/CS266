import numpy as np
import matplotlib.pyplot as plt

def tinkerbell(x,y, c_1 = -0.3, c_2=-0.6, c_3=2, c_4=0.5):
    return (x*x - y*y + c_1 * x + c_2 * y,
            2*x*y + c_3*x + c_4 * y)


for c_4 in [0.4, 0.5, 0.6]:
    values = [(0.1, 0.1)]

    for i in range(1000):
        values.append(tinkerbell(*values[-1], c_4=c_4))


    x, y = zip(*values)

    plt.plot(x,y, linewidth=0.2)
    plt.title(r"Tinkerbell map | $c_4$: {}".format(c_4))
    plt.xlim((-0.25, 0.25))
    plt.ylim((-0.5, 0.3))
    plt.savefig("a9_q4_c_4_{}.pdf".format(c_4))
    plt.show()


N = 1000
values = [(0.1, 0.1)]

for i in range(N):
    values.append(tinkerbell(*values[-1], c_1=0.9))

x, y = zip(*values)
plt.plot(x,y, linewidth=0.1)
plt.title(r"Tinkerbell map | $c_1$: {}".format(0.9))
plt.savefig("a9_q5.pdf")
plt.show()


