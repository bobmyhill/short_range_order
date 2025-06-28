import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.constants import gas_constant as R


def helmholtz_energy_square_lattice(T, J=1):
    beta = 1.0 / (R * T)
    K = beta * J
    tanh2K = np.tanh(2 * K)
    sech2K = 1.0 / np.cosh(2 * K)
    kappa = 0.5 * tanh2K * sech2K

    integral, _ = dblquad(
        lambda w1, w2: np.log(1 - 4 * kappa * np.cos(w1) * np.cos(w2)),
        0, np.pi,  # Limits for w2
        lambda _: 0, lambda _: np.pi  # Limits for w1
    )

    ln_lambda = np.log(2 * np.cosh(2 * K)) + integral / (2 * np.pi**2)

    return -R * T * ln_lambda


if __name__ == "__main__":

    T_values = np.linspace(0.01, 1.0, 200)
    F_values = np.array([helmholtz_energy_square_lattice(T) for T in T_values])
    S_values = -np.gradient(F_values, T_values)

    plt.figure(figsize=(8, 5))
    plt.plot(T_values, S_values/R, label=r'$f(T)$ from Onsager')
    plt.xlabel('Temperature')
    plt.ylabel('Entropy')
    plt.title('Onsager Solution for entropy on a square lattice')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
