from cmath import sqrt
from random import random


def parabolas(f, x0, x1, x2, tol=1e-9, maxiter=100, return_xs=False):
    xs = [x0, x1, x2]
    fs = [f(x0), f(x1), f(x2)]

    ii = 0
    while abs(fs[-1]) > tol and ii < maxiter:
        ii += 1
        a = b = c = 0
        for i in range(3):
            q = fs[i] / ((xs[i] - xs[i - 1]) * (xs[i] - xs[i - 2]))
            a += q
            b -= (xs[i - 1] + xs[i - 2]) * q
            c += (xs[i - 1] * xs[i - 2]) * q

        D = b * b - 4 * a * c
        r1 = (-b + sqrt(D)) / (2 * a)
        r2 = (-b - sqrt(D)) / (2 * a)

        if abs(r1 - xs[-1]) < abs(r2 - xs[-1]):
            xs = xs[1:] + [r1]
            fs = fs[1:] + [f(r1)]
        else:
            xs = xs[1:] + [r2]
            fs = fs[1:] + [f(r2)]

    if return_xs:
        return xs
    else:
        return xs[-1]


def multiple_roots(f, n_roots=1, x0=None, x1=None, x2=None, tol=1e-9):
    xs = [x0, x1, x2]
    for i in range(3):
        if xs[i] is None:
            xs[i] = random()

    roots = []

    while len(roots) < n_roots:
        def f_mod(x):
            res = f(x)
            for root in roots:
                res /= (x - root)
            return res

        xs_new = parabolas(f_mod, *xs, maxiter=5, return_xs=True)
        roots.append(parabolas(f, *xs_new, tol=tol))

    return roots


if __name__ == '__main__':
    from cmath import cos
    import matplotlib.pyplot as plt

    def f(x):
        # roots: 2pi * k ± log(3 + sqrt(8)) * j
        return cos(x) - 3

    def p(x):
        # roots: 0, 0, 1, 2, 5, ±j, 0.5 ± 0.5 * sqrt(3) * j
        return x * x * (x - 1) * (x - 2) * (x - 5) * (x * x + 1) * (x * x - x + 1)

    single_root = parabolas(f, 0, 1, 2)
    roots = multiple_roots(p, n_roots=9)

    plt.scatter([x.real for x in roots], [x.imag for x in roots])
    plt.grid()
    plt.show()
    print(single_root)
    print(roots)
