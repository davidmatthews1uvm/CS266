import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

n = 1000
pixels = ti.field(dtype=float, shape=(n , n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.kernel
def clear():
    for i,j in pixels:
        pixels[i,j] = 0.0


@ti.kernel
def paint(a: float, b: float, iter_cnt: int):
    for i, j in pixels:  # Parallized over all pixels
        c = [a, b] # ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 0.5, j / n - 0.5]) * 4
        is_origin = ((i)/n == 0.5 and (j)/n == 0.5)

        iterations = 0
        while z.norm() < 20 and iterations < iter_cnt:
            z = complex_sqr(z) + c
            iterations += 1
        if is_origin:
            print("c:", c[0], "+", c[1], "i")

            for print_idx in ti.static(range(8)):
                print(print_idx, ":", z[0], "+", z[1], "i")
                z = complex_sqr(z) + c

        pixels[i, j] = 1 - iterations/iter_cnt



c_vals = [(-0.5, 0.3), (0, 1), (-1, 0), (0, 1.1)]

for a, b in c_vals:
    clear()
    paint(a, b, 200)
    plt.imshow(pixels.to_numpy())
    plt.title(r"$c = {} + i{}$".format(a,b))
    plt.savefig("Assignment_7_{:.2f}_i{:.2f}_julia.pdf".format(a,b), bbox_inches="tight", pad_inches=0)
    plt.show()
