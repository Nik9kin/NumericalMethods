import numpy as np


def polynomial(xs, fs):
    n = len(xs)
    ids = np.arange(n)
    np.random.shuffle(ids)
    xs = np.array(xs)[ids]
    fs = np.array(fs)[ids]

    fss = np.diag(fs)
    for k in range(1, n):
        for i in range(n - k):
            fss[i, i + k] = (fss[i, i + k - 1] - fss[i + 1, i + k]) / (xs[i] - xs[i + k])

    xs = np.array(xs)[:-1]
    xs = xs.reshape(-1, 1)
    f0 = fss[0].reshape(-1, 1)

    def p(x):
        return np.sum(np.cumprod(x - xs, axis=0) * f0[1:], axis=0) + f0[0]

    return p


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f = np.abs
    for n_xs in [5, 11, 21, 51, 101]:
        x1 = np.linspace(-1, 1, n_xs)
        p1 = polynomial(x1, f(x1))

        x2 = np.cos(np.linspace(0, np.pi, n_xs))
        p2 = polynomial(x2, f(x2))

        xs = np.linspace(-1, 1, 10001)
        plt.figure(figsize=(10, 5))
        plt.plot(xs, f(xs), label=r"|$\cdot$|")
        plt.plot(xs, p1(xs), label="poly by regular grid")
        plt.plot(xs, p2(xs), label="poly by cosine grid")
        plt.title(f"{n_xs} dots for interpolation")
        plt.legend()
        plt.grid()
        # plt.xlim(-1.2, 1.2)
        plt.ylim(-0.1, 1.1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    f = np.sin
    for n_xs in [5, 11, 21, 31, 51, 71]:
        x1 = np.linspace(0, 10 * np.pi, n_xs)
        p1 = polynomial(x1, f(x1))

        x2 = np.random.uniform(10 * np.pi, size=n_xs - 2)
        x2 = np.hstack((x2, np.array([0.0, 10 * np.pi])))
        p2 = polynomial(x2, f(x2))

        xs = np.linspace(0, 10 * np.pi, 10001)
        plt.figure(figsize=(10, 5))
        plt.plot(xs, f(xs), label=r"$\sin(\cdot)$")
        plt.plot(xs, p1(xs), label="poly by regular grid")
        plt.plot(xs, p2(xs), label="poly by random (uniform) grid")
        plt.title(f"{n_xs} dots for interpolation")
        plt.legend()
        plt.grid()
        # plt.xlim(-1.2, 1.2)
        plt.ylim(-10.1, 10.1)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
