import numpy as np
import matplotlib.pyplot as plt

def tinkerbell(x,y, c_1 = -0.3, c_2=-0.6, c_3=2, c_4=0.5):
    return (x*x - y*y + c_1 * x + c_2 * y,
            2*x*y + c_3*x + c_4 * y)

def henon(x,y, a=1.4, b=0.3):
    return (a - x**2 + b * y, x)

def ikeda(x,y, r, c_1, c_2, c_3):
    tau = c_1 - c_3/(1+x*x + y*y)
    return (r + c_2 * (x * np.cos(tau)- y * np.sin(tau)),
                c_2 * (x * np.sin(tau)+ y * np.cos(tau)))

N = 10000
values = np.zeros((N+1, 2))
values[0, :] = np.random.random(2)
lyapunov_ests = np.zeros((N, 2))
W = np.eye(2)


# henon map
a = 1.4
b = 0.3

for i in range(1, N+1):
    x,y =  henon(values[i - 1, 0], values[i - 1, 1], a=a, b=b)
    values[i, :] = (x,y)
    J = np.array([[-2 * x, b], [1, 0]])
    W, r = np.linalg.qr(np.dot(J, W))
    lyapunov_ests[i-1, :] =   r[0,0], r[1,1]
print("henon map lyapunov exponents")
lyapunov_exps = np.mean(np.log(np.abs(lyapunov_ests)), axis=0)
print(lyapunov_exps)



# values[:, :] = 0
# values[0, :] = np.random.random(2)
# lyapunov_ests[:, L] = 0
# W = np.eye(2)


# # ikeda map
# r = 1
# c_1 = 0.4
# c_2 = 0.9
# c_3 = 6

# for i in range(1, N+1):
#     x,y =  ikeda(values[i - 1, 0], values[i - 1, 1], r=r, c_1=c_1, c_2=c_2, c_3=c_3)
#     values[i, :] = (x,y)
#     # J = np.array([[-2 * x, b], [1, 0]])
#     W, r = np.linalg.qr(np.dot(J, W))
#     lyapunov_ests[i-1, :] =   r[0,0], r[1,1]
# print("tinkerbell map lyapunov exponents")
# lyapunov_exps = np.mean(np.log(np.abs(lyapunov_ests)), axis=0)
# print(lyapunov_exps)


values[:, :] = 0
values[0, :] = [0.1, 0.1]
lyapunov_ests[:, :] = 0
W = np.eye(2)
# tinkerbell map
c_1 = -0.3
c_2=-0.6
c_3=2
c_4=0.5

for i in range(1, N+1):
    x,y =  tinkerbell(values[i - 1, 0], values[i - 1, 1], c_1 = c_1, c_2=c_2, c_3=c_3, c_4=c_4)
    values[i, :] = (x,y)
    J = np.array([[2 * x + c_1, 2*y + c_3], [-2*y + c_2, 2*x + c_4]])
    W, r = np.linalg.qr(np.dot(J, W))
    lyapunov_ests[i-1, :] =  r[0,0], r[1,1]

print("Tinkerbell map lyapunov exponents")
lyapunov_exps = np.mean(np.log(np.abs(lyapunov_ests)), axis=0)
print(lyapunov_exps)


values[:, :] = 0
values[0, :] = [0.1, 0.1]
lyapunov_ests[:, :] = 0
W = np.eye(2)
# tinkerbell map
c_1 = 0.9
c_2=-0.6
c_3=2
c_4=0.5

for i in range(1, N+1):
    x,y =  tinkerbell(values[i - 1, 0], values[i - 1, 1], c_1 = c_1, c_2=c_2, c_3=c_3, c_4=c_4)
    values[i, :] = (x,y)
    J = np.array([[2 * x + c_1, 2*y + c_3], [-2*y + c_2, 2*x + c_4]])
    W, r = np.linalg.qr(np.dot(J, W))
    lyapunov_ests[i-1, :] =  r[0,0], r[1,1]

print("Tinkerbell map lyapunov exponents (c_1 = 0.9)")
lyapunov_exps = np.mean(np.log(np.abs(lyapunov_ests)), axis=0)
print(lyapunov_exps)
# plt.plot(values[:, 0], values[:, 1])
# plt.show()
