import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from scipy.constants import gas_constant as R

# === Load data ===
data = np.loadtxt("input_data/enthalpies.dat")
T = data[:, 0] / (R * 4.0)  # Temperature
H = data[:, 1]  # Enthalpy (e.g., in J/mol or eV/atom)

# === Define breakpoint for phase transition ===
T_break = 1.8593599 / (R * 4.0)

# === Split data at the break ===
left_mask = T <= T_break
right_mask = T >= T_break

T_left = T[left_mask]
H_left = H[left_mask]
T_right = T[right_mask]
H_right = H[right_mask]

# === Fit splines ===
spline_left = PchipInterpolator(T_left, H_left)  # Monotonic fit
spline_right = UnivariateSpline(T_right, H_right, k=3, s=1.0)  # Smooth fit

# === Evaluation grids ===
T_dense_left = np.linspace(np.min(T_left), T_break, 500)
T_dense_right = np.linspace(T_break, np.max(T_right), 500)


H_dense_left = spline_left(T_dense_left)
H_dense_right = spline_right(T_dense_right)


# === Derivatives of enthalpy ===
dH_dT_left = spline_left.derivative()(T_dense_left)
dH_dT_right = spline_right.derivative()(T_dense_right)

# === Avoid division by zero ===
epsilon = 1e-8
T_safe_left = np.where(T_dense_left == 0, epsilon, T_dense_left)
T_safe_right = np.where(T_dense_right == 0, epsilon, T_dense_right)

# === Compute integrand dH/dT / T ===
integrand_left = dH_dT_left / T_safe_left
integrand_right = dH_dT_right / T_safe_right

# === Integrate to compute entropy ===
S_left = cumulative_trapezoid(integrand_left, T_dense_left, initial=0)
S_right = cumulative_trapezoid(integrand_right, T_dense_right, initial=0)

# === Compute entropy jump from enthalpy jump ===
H_before = spline_left(T_break)
H_after = spline_right(T_break)
delta_H = H_after - H_before
delta_S = delta_H / T_break

# === Correct the entropy shift on the right ===
S_right += S_left[-1] + delta_S  # This includes the entropy jump

# === Merge left and right pieces ===
T_full_A3B = np.concatenate((T_dense_left, T_dense_right[1:]))
S_full_A3B = np.concatenate((S_left, S_right[1:]))
H_full_A3B = np.concatenate((H_dense_left, H_dense_right[1:]))

if __name__ == "__main__":
    print(f"--- Jump at T = {T_break:.7f} ---")
    print(f"ΔH = {delta_H:.6f} (same units as input H)")
    print(f"ΔS = {delta_S:.6f} (same units per K)")

    # === Plot ===
    plt.figure(figsize=(10, 5))
    plt.scatter(T, H)
    plt.plot(T_full_A3B, H_full_A3B, label="Entropy $S(T)$", color="indigo")
    plt.axvline(
        T_break, color="gray", linestyle="--", label=f"Break at T = {T_break:.7f}"
    )

    plt.xlabel("Temperature (T)")
    plt.ylabel("Entropy (S)")
    plt.title("Entropy from Enthalpy with First-Order Phase Transition")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
