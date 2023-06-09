# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import PIL.Image as im
import sympy as sp
import random


def gen_cplx_grid(xmin, xmax, nx, ymin, ymax, ny):
    xstep = np.linspace(xmin, xmax, nx)
    ystep = np.linspace(ymin, ymax, ny)
    return xstep[:, None] + 1j * ystep


def newton_iter(x, roots, size, f, df):
    while min(list(map(lambda root: abs(root - x), roots))) > size:
        x -= (f(x) / df(x))
    return x


def colorize_point(point, roots, colors):
    return colors[min_index(list(map(lambda root: abs(root - point), roots)))]


def min_index(temp):
    return temp.index(min(temp))


def make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors):
    grid = gen_cplx_grid(xmin, xmax, nx, ymin, ymax, ny)
    image = im.new("RGB", (nx, ny))
    px = image.load()
    for i in range(nx):
        for j in range(ny):
            px[i, j] = colorize_point(newton_iter(grid[i, j], roots, ((xmax - xmin) / 100), f, df), roots, colors)
    image.save("/Users/bbaptist/PycharmProjects/NewtonFractals/bg/im" + str(counter) + ".png")
    print(counter)


def roots_to_funcs(roots):
    if len(roots) == 0:
        raise Exception("!!NO ROOTS!!")
    x = sp.symbols('x')
    f = sp.simplify(np.prod(list(map(lambda root: x - root, roots))))
    df = sp.diff(f, x)
    return sp.lambdify(x, f), sp.lambdify(x, df)


def main():
    nx = 2560
    ny = 1600
    xmin = nx * -1
    xmax = nx
    ymin = ny * -1
    ymax = ny
    counter = 43
    while True:
        roots = []
        colors = []
        #colors = [(254, 67, 101), (252, 157, 154), (249, 205, 173), (200, 200, 169), (131, 175, 155)]
        for i in range(random.randrange(4,7)):
            roots.append(random.uniform(ymin, ymax) * 1j + random.uniform(xmin, xmax))
            colors.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        f, df = roots_to_funcs(roots)
        counter += 1
        make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
main()
