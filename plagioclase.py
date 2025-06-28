import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.constants import gas_constant as R
from A3B_data import H_full_A3B, S_full_A3B, T_full_A3B

def make_transparent(img, white_thresh=0.95):
    if img.shape[2] == 3:  # No alpha channel yet
        alpha = np.ones((img.shape[0], img.shape[1]))
        white_mask = np.all(img > white_thresh, axis=-1)
        alpha[white_mask] = 0  # Make white pixels transparent
        img = np.dstack((img, alpha))
    else:
        white_mask = np.all(img[:, :, :3] > white_thresh, axis=-1)
        img[:, :, 3][white_mask] = 0

    return img


def excess_vanLaar(p, alpha, w):
    """
    Compute excess using van Laar-type asymmetric model:
        excess = alpha.T @ p * (phi.T @ W @ phi)

    Parameters:
    - p: (n,) array-like, mole or site fractions
    - alpha: (n,) array-like, van Laar asymmetry parameters
    - w: (n, n) array-like, raw interaction parameters w_ij (symmetric)

    Returns:
    - excess: float, excess property
    """

    # Normalized phi_i = alpha_i * p_i / sum(alpha * p)
    phi = ((alpha * p).T / np.sum(alpha * p, axis=1)).T

    # Compute W_ij = 2 * w_ij / (alpha_i + alpha_j)
    alpha_sum = alpha[:, None] + alpha[None, :]

    W = 2 * w / alpha_sum
    # Ensure symmetry
    W = (np.triu(W, 1) + np.triu(W, 1).T)/2.


    # Calculate G_excess
    i = np.einsum("ij, jk, ki->i", phi, W, phi.T)
    excess = np.dot(alpha, p.T) * i
    return excess




# Create figure and single subplot in a list
fig = plt.figure(figsize=(6, 4))  # you can adjust the figure size
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

# Display image with extent
img = mpimg.imread("figures/Dubacq_2022_Fig_9a.png")
ax[0].imshow(make_transparent(img), extent=[-0.009, 1.009, -0.15, 30.03], aspect='auto')
img = mpimg.imread("figures/Dubacq_2022_Fig_8a.png")
ax[1].imshow(make_transparent(img), extent=[0, 5000, 1., 4.2], aspect='auto')

# 50:50
img = mpimg.imread("figures/Dubacq_2022_Fig_8b.png")
ax[2].imshow(make_transparent(img), extent=[0, 5000, 0., 25], aspect='auto')

img = mpimg.imread("figures/Oates_et_al_1999_Figure_2.png")
ax[2].imshow(make_transparent(img), extent=[0, 5000, 0., 4.*R*0.6], aspect='auto')

ax[2].set_ylim(0., 25.)


img = mpimg.imread("figures/Dubacq_2022_Fig_8b.png")
ax[3].imshow(make_transparent(img), extent=[0, 5000, 0., 25], aspect='auto')

ax[3].plot(T_full_A3B*14000., S_full_A3B)
# 25:75

x_An = np.linspace(1.e-6, 1.-1.e-6, 101)
x_Ab = 1. - x_An

p_Al = (2. * x_An + 3. * x_Ab)/4.
p_Si = 1. - p_Al

S_A = -R * (x_An*np.log(x_An) + x_Ab*np.log(x_Ab))
S_T = -R * (p_Al*np.log(p_Al) + p_Si*np.log(p_Si))

# an then ab
alphas = np.array([0.550, 0.674])
w = np.array([[0., 9.35], [0., 0.]])
p = np.array([x_An, x_Ab]).T
Ss = excess_vanLaar(p, alphas, w)


ax[0].plot(x_An, Ss, linestyle=":")


ax[0].plot(x_An, S_A, linestyle=":")
ax[0].plot(x_An, S_T, linestyle=":")
ax[0].plot(x_An, 4.*S_T, linestyle=":")
ax[0].plot(x_An, S_A + 4.*S_T, linestyle=":")
ax[0].plot(x_An, S_A + S_T + Ss, label="HGP2022")

# Set axis limits
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 30)

ax[0].set_xlabel("X(Ca)")
ax[0].set_ylabel("S config")
ax[0].set_title("Dubacq (2022) Fig. 9a (0<T<2000)")
ax[0].legend()
plt.show()