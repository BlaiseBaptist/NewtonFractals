# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import PIL.Image as im
import re
import sympy as sp
import random


# Use a breakpoint in the code line below to debug your script.
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
    image.save("/Users/bbaptist/PycharmProjects/NewtonFractals/2k/2k" + str(counter) + ".png")
    print(counter)


def roots_to_funcs(roots):
    if len(roots) == 0:
        raise Exception("!!NO ROOTS!!")
    x = sp.symbols('x')
    f = sp.simplify(np.prod(list(map(lambda root: x - root, roots))))
    df = sp.diff(f, x)
    return sp.lambdify(x, f), sp.lambdify(x, df)


def main():
    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2
    ny = 2048
    nx = 2048
    counter = 46
    rand = True
    while rand:
        roots = []
        colors = []
        for i in range(random.randrange(3, 10)):
            roots.append(random.uniform(xmin, xmax) * 1j + random.uniform(xmin, xmax))
            colors.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        f, df = roots_to_funcs(roots)
        counter += 1
        make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors)
    finder = re.compile(r"(-?\d+)[xy]")
    roots = [1,1j+1,-1,]
    colors = [(255,0,0),(0,255,0),(0,0,255)]
    f, df = roots_to_funcs(roots)
    counter +=1
    make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors)
    while not rand:
        result1 = re.findall(finder, input("top right corner"))
        if len(result1) >= 2:
            result2 = re.findall(finder, input("bottom left corner"))
            if len(result2) >= 2:
                xmin = int(result2[0]) * xmin / 100 * -1
                ymin = int(result2[1]) * ymin / 100 * -1
                xmax = int(result1[0]) * xmax / 100
                ymax = int(result1[1]) * ymax / 100
                counter +=1
                make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors)


# \d(?:\D*)([+]\d+j)*
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
main()
