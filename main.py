import numpy as np
import PIL.Image as im
import sympy as sp
import random
import os
import time


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


def make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors, path, start_time):
    grid = gen_cplx_grid(xmin, xmax, nx, ymin, ymax, ny)
    image = im.new("RGB", (nx, ny))
    px = image.load()
    for i in range(nx):
        for j in range(ny):
            px[i, j] = colorize_point(newton_iter(grid[i, j], roots, ((xmax - xmin) / 100), f, df), roots, colors)
    image.save(path + "/im" + str(counter) + ".png")
    run_time = time.time() - start_time
    px_time = (run_time * 1000000) / (ny * nx)
    return "made image " + str(counter) + " in " + str(round(run_time, 5)) + "s " + str(
        round(px_time, 5)) + "us per pixel"


def roots_to_funcs(roots):
    if len(roots) == 0:
        raise Exception("!!NO ROOTS!!")
    x = sp.symbols('x')
    f = sp.simplify(np.prod(list(map(lambda root: x - root, roots))))
    df = sp.diff(f, x)
    return sp.lambdify(x, f), sp.lambdify(x, df)


def main():
    nx = 200
    ny = 200
    xmin, xmax = nx * -1, nx
    ymin, ymax = ny * -1, ny
    rand = True
    path = "/Users/bbaptist/PycharmProjects/newton_fractals/tests"
    files = list(map(lambda name: int(name[2:-4]), os.listdir(path)))
    if files:
        counter = max(files)
    else:
        counter = 0
    while rand:
        roots = []
        colors = []
        for i in range(random.randrange(4, 5)):
            roots.append(random.uniform(ymin, ymax) * 1j + random.uniform(xmin, xmax))
            colors.append((random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))
        roots = [-1.0,-2.0,-3.0]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        print(roots)
        xmax, ymax = 10.0, 10.0
        xmin, ymin = -10.0, -10.0
        f, df = roots_to_funcs(roots)
        counter += 1
        print(make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, counter, colors, path, time.time()))
        f1 = f(1.0+1.0j)
        df1 = df(1.0+1.0j)
        print(f1,df1,f1/df1)
        print((1+1j)/(1+1j))
        return True
    if not rand:
        # colors = [(254, 67, 101), (252, 157, 154), (249, 205, 173), (200, 200, 169), (131, 175, 155)]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        roots = [256, -256, 256j]
        f, df = roots_to_funcs(roots)
        print(make_im(xmin, xmax, nx, ymin, ymax, ny, roots, f, df, 0, colors, path, time.time()))


main()
