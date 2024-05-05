import numpy as np


def euler(x0, f, t0=0.0, dt=1e-3, n_steps=10_000):
    tx = np.hstack((t0, x0))
    traj = np.zeros((n_steps + 1, tx.shape[0]))
    traj[0] = tx
    traj[:, 0] = np.linspace(t0, t0 + n_steps * dt, n_steps + 1, endpoint=True)

    for i in range(n_steps):
        traj[i + 1, 1:] = traj[i, 1:] + np.array(f(*traj[i])) * dt

    return traj


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def f1(x, y, y_prime):
        # equation for y = sin(x)
        # y" = -y
        # general solution is y = A * sin(x) + B * cos(x)

        return y_prime, -y

    T = 20
    xs = np.linspace(0.0, T, 10_001)
    ys = np.sin(xs)

    for dt in (0.1, 0.01, 0.001):
        sol = euler((0, 1), f1, dt=dt, n_steps=int(np.rint(T / dt)))
        plt.plot(sol[:, 0], sol[:, 1], label=f"dt = {dt}")

    plt.plot(xs, ys, label="exact")
    plt.title('y = sin(x), y" = -y')
    plt.legend()
    plt.grid()
    plt.show()

    T = 6 * np.pi
    ts = np.linspace(0.0, T, 10_001)
    xs = np.cos(ts)
    ys = np.sin(ts)

    for dt in (0.1, 0.01, 0.001):
        sol = euler((0, 1), f1, dt=dt, n_steps=int(np.rint(T / dt)))
        plt.plot(sol[:, 2], sol[:, 1], label=f"dt = {dt}")

    plt.plot(xs, ys, label="exact")
    plt.title("(x, y) = (cos(t), sin(t)), (x', y') = (-y, x)")
    plt.legend()
    plt.grid()
    plt.show()


    def f2(x, y):
        # equation for y = exp(x)
        # y' = y
        # general solution is y = A * exp(x)

        return y

    T = 3
    xs = np.linspace(0.0, T, 1001)
    ys = np.exp(xs)

    for dt in (0.1, 0.01, 0.001):
        sol = euler(1, f2, dt=dt, n_steps=int(np.rint(T / dt)))
        plt.plot(sol[:, 0], sol[:, 1], label=f"dt = {dt}")

    plt.plot(xs, ys, label="exact")
    plt.title("y = exp(x), y' = y")
    plt.legend()
    plt.grid()
    plt.show()


    def f3(t, x, y, z, s=10, r=28, b=2.667):
        x_prime = s * (y - x)
        y_prime = r * x - y - x * z
        z_prime = x * y - b * z
        return x_prime, y_prime, z_prime

    sol = euler((0.1, 0.2, 10), f3, dt=0.001, n_steps=50_000)
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(sol[:, 1], sol[:, 2], sol[:, 3], lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()
