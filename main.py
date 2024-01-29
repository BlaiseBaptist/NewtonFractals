import numpy as np
import PIL.Image as im
import sympy as sp
import random
import os


def gen_cplx_grid(xmin, xmax, nx, ymin, ymax, ny):
    xstep = np.linspace(xmin, xmax, nx)
    ystep = np.linspace(ymin, ymax, ny)
    return xstep[:, None] + 1j * ystep


def newton_iter(x, roots, size, f, df):
    i = 1
    while min(list(map(lambda root: abs(root - x), roots))) > size:
        x -= (f(x) / df(x))
        i += 1
    return x, i


def colorize_point(data, roots, colors, invert=False, shade=False):
    color = colors[min_index(list(map(lambda root: abs(root - data[0]), roots)))]
    if shade:
        if invert:
            return tuple(map(lambda val: int((val / 8) * data[1]), color))
        return tuple(map(lambda val: int((val * 1.6) / (np.log(data[1]) + 1)), color))
    return color


def min_index(temp):
    return temp.index(min(temp))


def make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors, path):
    grid = gen_cplx_grid(xmin, xmax, nx, ymin, ymax, ny)
    image = im.new("RGB", (nx, ny))
    px = image.load()
    for i in range(nx):
        for j in range(ny):
            px[i, j] = colorize_point(newton_iter(grid[i, j], roots, ((xmax - xmin) / 100), f, df), roots, colors)
    image.save(path + "/im" + str(counter) + ".png")
    print(counter)


def roots_to_funcs(roots):
    if len(roots) == 0:
        raise Exception("!!NO ROOTS!!")
    x = sp.symbols('x')
    f = sp.simplify(np.prod(list(map(lambda root: x - root, roots))))
    df = sp.diff(f, x)
    return sp.lambdify(x, f), sp.lambdify(x, df)


def main():
    nx = 4096
    ny = 4096
    xmin = nx * -1
    xmax = nx
    ymin = ny * -1
    ymax = ny
    rand = True
    path = "/Users/bbaptist/PycharmProjects/NewtonFractals/4k"
    files = list(map(lambda name: int(name[2:-4]), os.listdir(path)))
    if files:
        counter = max(files)
    else:
        counter = 0
    print("working on:", counter)
    while rand:
        roots = []
        colors = []
        for i in range(random.randrange(4, 7)):
            roots.append(random.uniform(ymin, ymax) * 1j + random.uniform(xmin, xmax))
            colors.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        f, df = roots_to_funcs(roots)
        counter += 1
        make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors, path)
    if not rand:
        # colors = [(254, 67, 101), (252, 157, 154), (249, 205, 173), (200, 200, 169), (131, 175, 155)]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        roots = [256, -256, 256j]
        f, df = roots_to_funcs(roots)
        make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, 0, colors, path)


main()
