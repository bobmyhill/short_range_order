import numpy as np
from itertools import product
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.constants import gas_constant as R
import sympy as sp
from burnman.utils.reductions import independent_row_indices
from scipy.optimize import root
import matplotlib.image as mpimg
from matplotlib.lines import Line2D


cmap = cm.cividis

colour = [
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
]


def build_weighted_flat_interaction_matrix(
    site_species, interaction_matrix, bonds, n_bonds
):
    """
    Build a symmetric weighted flattened interaction matrix for undirected bonds.

    This matrix encodes pairwise interactions between all allowed species
    at all sites, weighted by bond multiplicities. The matrix is symmetric
    because bonds are undirected and interactions are mutual.

    :param site_species: List of allowed species at each site.
                         Example: [[0, 1], [0, 2], [1]]
    :type site_species: list[list[int]]

    :param interaction_matrix: Symmetric matrix of pairwise interaction energies
                               between species. Shape: (n_species, n_species).
    :type interaction_matrix: np.ndarray

    :param bonds: Array of bonded site pairs.
                  Shape: (n_bonds, 2). Each entry [i, j] indicates a bond
                  between site i and site j.
    :type bonds: array-like of int

    :param n_bonds: Bond weights or multiplicities for each bond.
                    Shape: (n_bonds,).
                    These weight the contribution of each bond to the interaction matrix.
    :type n_bonds: array-like of float

    :return: Symmetric weighted flattened interaction matrix.
             Shape: (total_site_species, total_site_species), where
             total_site_species = sum of allowed species counts over all sites.
    :rtype: np.ndarray
    """

    # Step 1: Determine how many allowed species per site
    # e.g. site_species = [[0,1],[0,2],[1]] => species_counts = [2,2,1]
    species_counts = [len(s) for s in site_species]
    total_site_species = sum(species_counts)

    # Step 2: Compute offsets for flattening site-species indices
    # offsets[i] = starting index in flattened vector for site i
    # For example, offsets = [0, 2, 4] means site 0 species indices go from 0..1,
    # site 1 from 2..3, site 2 from 4..4 (single species)
    offsets = np.cumsum([0] + species_counts[:-1])

    # Step 3: Create a lookup dictionary to map (site, species) pairs to flat indices
    # For quick indexing into flattened interaction matrix later
    species_to_flat = {}
    for site, species_list in enumerate(site_species):
        for idx, species in enumerate(species_list):
            flat_index = offsets[site] + idx
            species_to_flat[(site, species)] = flat_index
            # Example: (site=1, species=2) -> 3 (if offsets[1]=2 and idx=1)

    # Step 4: Create the flattened interaction matrix.
    # Iterate over each bond between sites
    # bonds is an array like [[0,1], [1,2], ...]
    flat_interaction = np.zeros((total_site_species, total_site_species))
    for b_idx, (site1, site2) in enumerate(bonds):
        weight = n_bonds[b_idx]  # bond multiplicity/weight for this bond
        if len(n_bonds) > 10:
            print(b_idx, (site1, site2))

        # Retrieve the allowed species lists for the two bonded sites
        species_list1 = site_species[site1]
        species_list2 = site_species[site2]

        # For every pair of species on the two bonded sites,
        # add the weighted interaction energy to the flattened matrix
        for sp1 in species_list1:
            idx1 = species_to_flat[(site1, sp1)]  # flattened index for (site1, sp1)
            for sp2 in species_list2:
                idx2 = species_to_flat[(site2, sp2)]  # flattened index for (site2, sp2)

                # Interaction energy from the base matrix scaled by bond weight
                interaction_val = weight * interaction_matrix[sp1, sp2]
                # Add the interaction energy symmetrically to ensure
                # that interaction energy between two site-species pairs
                # is counted equally regardless of ordering.
                flat_interaction[idx1, idx2] += interaction_val
                flat_interaction[idx2, idx1] += interaction_val

    return flat_interaction


def compute_cluster_site_species_occupancies(site_species):
    """
    Generate all cluster configurations and one-hot encodings of site-specific species occupancies.

    :param site_species: List of allowed species per site, e.g., [[0, 1], [0, 2]].
    :type site_species: list[list[int]]

    :return: One-hot encoded occupancy vector per configuration
        (n_clusters, sum(len(s) for s in site_species))
    :rtype: numpy.ndarray
    """

    # Step 1: Cartesian product to get all cluster occupancies
    cluster_occupancies = np.array(list(product(*site_species)))
    n_configs, n_sites = cluster_occupancies.shape

    # Step 2: Flattened size and cumulative offsets
    species_counts = [len(s) for s in site_species]
    offsets = np.cumsum([0] + species_counts[:-1])  # starting index of each site block

    # Step 3: Build lookup table: species -> index within each site
    species_to_index = {}
    for site, species_list in enumerate(site_species):
        for i, species in enumerate(species_list):
            species_to_index[(site, species)] = i

    # Step 4: Compute flattened indices for 1-hot encoding
    flat_indices = np.array(
        [
            offsets[site] + species_to_index[(site, cluster_occupancies[i, site])]
            for i in range(n_configs)
            for site in range(n_sites)
        ]
    )

    # Step 5: Construct the binary array
    total_length = sum(species_counts)
    cluster_site_species_occupancies = np.zeros((n_configs, total_length), dtype=int)
    row_indices = np.repeat(np.arange(n_configs), n_sites)
    cluster_site_species_occupancies[row_indices, flat_indices] = 1

    return cluster_site_species_occupancies


def compute_cluster_properties(site_species, interaction_matrix, bonds, n_bonds):
    """
    Compute interaction energies for all possible cluster configurations using
    a weighted, flattened interaction matrix and site-species occupancies.

    Each configuration represents one allowed assignment of species to each site,
    constrained by the allowed species at each site. The energy of a configuration
    is calculated by summing all pairwise interactions over bonded sites, weighted
    by bond multiplicities.

    :param site_species: Allowed species at each site.
                         Example: [[0, 1], [0, 2], [1]] means:
                         - site 0 can be species 0 or 1
                         - site 1 can be species 0 or 2
                         - site 2 can be species 1
    :type site_species: list[list[int]]

    :param interaction_matrix: Symmetric pairwise interaction energy matrix.
                               Shape: (n_species, n_species)
    :type interaction_matrix: np.ndarray

    :param bonds: Array of bonded site pairs.
                  Each element is a pair [i, j] representing a bond between site i and site j.
                  Shape: (n_bonds, 2)
    :type bonds: array-like of int

    :param n_bonds: Array of weights or multiplicities for each bond.
                    Shape: (n_bonds,)
    :type n_bonds: array-like of float

    :return: A tuple containing:
             - cluster_energies: Total energy of each configuration (sum over all bonds).
               Shape: (n_clusters,)
             - cluster_site_species_occupancies: One-hot encoded array of cluster configurations.
               Shape: (n_clusters, total_site_species)
             - mean_field_interactions: Energy contributions to each site-species entry.
               Shape: (n_clusters, total_site_species)
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Construct one-hot encoded cluster configurations
    cluster_site_species_occupancies = compute_cluster_site_species_occupancies(
        site_species
    )  # shape: (n_clusters, total_site_species)

    # Build flattened, symmetric interaction matrix with bond weights
    flat_interaction = build_weighted_flat_interaction_matrix(
        site_species, interaction_matrix, bonds, n_bonds
    )

    # Compute mean field energy per site-species entry for each cluster
    # shape: (n_clusters, total_site_species)
    mean_field_interactions = 0.5 * np.einsum(
        "ij,jk->ik", cluster_site_species_occupancies, flat_interaction
    )

    # Project mean-field energy back onto configuration to get scalar total energy
    # shape: (n_clusters,)
    cluster_energies = np.einsum(
        "ij,ij->i", cluster_site_species_occupancies, mean_field_interactions
    )

    return cluster_energies, cluster_site_species_occupancies, mean_field_interactions


def make_energy_histogram(cluster_site_species_occupancies, cluster_energies):
    # Determine average fraction of species 1 in each cluster
    frac_1 = cluster_site_species_occupancies[:, ::2].mean(
        axis=1
    )  # e.g., 0.0, 0.25, ..., 1.0
    unique_fracs = np.unique(frac_1)  # Composition groups

    # Define histogram bins for energy
    unique_vals = np.unique(cluster_energies)
    # Create bin edges halfway between unique values
    bin_edges = np.concatenate(
        (
            [unique_vals[0] - 0.5 * (unique_vals[1] - unique_vals[0])],
            (unique_vals[:-1] + unique_vals[1:]) / 2,
            [unique_vals[-1] + 0.5 * (unique_vals[-1] - unique_vals[-2])],
        )
    )

    # bin_centers are just the unique_vals themselves:
    bin_centers = unique_vals

    # Compute per-composition histograms
    hist_data = []
    for f in unique_fracs:
        mask = frac_1 == f
        counts, _ = np.histogram(cluster_energies[mask], bins=bin_edges)
        hist_data.append(counts)
    hist_data = np.array(hist_data)  # shape (num_compositions, num_bins)

    return unique_fracs, bin_centers, hist_data


def cluster_probabilities(
    mu_independent_clusters,
    temperature,
    effective_energies_clusters,
    clusters_as_independent_cluster_fractions,
):
    mu_clusters = clusters_as_independent_cluster_fractions.dot(mu_independent_clusters)
    p_clusters = np.exp((mu_clusters - effective_energies_clusters) / (R * temperature))
    return p_clusters


def effective_cluster_energies(
    independent_cluster_fractions,
    independent_cluster_occupancies,
    cluster_energies,
    mean_field_interactions,
    mean_field_fraction,
):
    ind = independent_cluster_occupancies
    site_species_occupancies = ind.T.dot(independent_cluster_fractions)
    return (
        1.0 - mean_field_fraction
    ) * cluster_energies + mean_field_fraction * mean_field_interactions.dot(
        site_species_occupancies
    )


def delta_independent_cluster_proportions(
    mu_independent_clusters,
    T,
    independent_cluster_fractions,
    independent_cluster_occupancies,
    cluster_energies,
    mean_field_interactions,
    mean_field_fraction,
    clusters_as_independent_cluster_fractions,
):
    E_clusters = effective_cluster_energies(
        independent_cluster_fractions,
        independent_cluster_occupancies,
        cluster_energies,
        mean_field_interactions,
        mean_field_fraction,
    )
    p_clusters = cluster_probabilities(
        mu_independent_clusters,
        T,
        E_clusters,
        clusters_as_independent_cluster_fractions,
    )
    return (
        p_clusters.dot(clusters_as_independent_cluster_fractions)
        - independent_cluster_fractions
    )


def find_mu(
    T,
    independent_cluster_fractions,
    independent_cluster_occupancies,
    cluster_energies,
    mean_field_interactions,
    mean_field_fraction,
    clusters_as_independent_cluster_fractions,
    guess=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
):
    sol = root(
        delta_independent_cluster_proportions,
        guess,
        args=(
            T,
            independent_cluster_fractions,
            independent_cluster_occupancies,
            cluster_energies,
            mean_field_interactions,
            mean_field_fraction,
            clusters_as_independent_cluster_fractions,
        ),
        method="hybr",
    )

    if not sol.success:
        sol = root(
            delta_independent_cluster_proportions,
            guess,
            args=(
                T,
                independent_cluster_fractions,
                independent_cluster_occupancies,
                cluster_energies,
                mean_field_interactions,
                mean_field_fraction,
                clusters_as_independent_cluster_fractions,
            ),
            method="Krylov",
        )

    assert sol.success
    return sol


class Cluster(object):
    def __init__(self):
        pass

    def print_clusters(self):
        for i in range(len(self.cluster_energies)):
            print(
                f"Cluster {self.cluster_site_species_occupancies[i]}: Energy = {self.cluster_energies[i]}"
            )
        print(
            f"{len(self.cluster_site_species_occupancies)} total clusters, {len(np.unique(self.cluster_site_species_occupancies, axis=0))} unique clusters"
        )

    def energy_histogram(self, ax):
        # Plot stacked bar histogram
        unique_fracs, bin_centers, hist_data = make_energy_histogram(
            self.cluster_site_species_occupancies, self.cluster_energies
        )
        bottom = np.zeros_like(bin_centers)
        norm = mcolors.Normalize(vmin=unique_fracs.min(), vmax=unique_fracs.max())

        for i, f in enumerate(unique_fracs):
            color = cmap(norm(f))
            ax.bar(
                bin_centers,
                hist_data[i],
                bottom=bottom,
                width=0.5,
                label=f"{f:.2f}",
                align="center",
                color=color,
            )
            bottom += hist_data[i]

        ax.set_xlabel("Cluster energy")
        ax.set_ylabel("Count")
        ax.legend(title="Composition")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0.0, bottom.max() * 1.1)

    def independent_cluster_fractions(self, site_fractions):
        return (
            np.linalg.lstsq(
                self.independent_cluster_occupancies.T.astype(float),
                site_fractions,
                rcond=0,
            )[0]
            .round(decimals=12)
            .T
        )

    def mu(self, temperature, independent_cluster_fractions, mean_field_fraction):

        sol = find_mu(
            temperature,
            independent_cluster_fractions,
            self.independent_cluster_occupancies,
            self.cluster_energies,
            self.mean_field_interactions,
            mean_field_fraction,
            self.clusters_as_independent_cluster_fractions,
            guess=self.mu_guess,
        )

        mu_independent_clusters = sol.x
        self.mu_guess = mu_independent_clusters
        return mu_independent_clusters


class SingleCluster(Cluster):
    def __init__(self, site_species, bonds, n_bonds, interaction_matrix):
        self.name = "single"
        self.species_counts = [len(s) for s in site_species]
        self.total_site_species = int(sum(self.species_counts))
        self.total_sites = len(site_species)

        self.bonds = bonds
        self.n_bonds = n_bonds

        # Calculates the energies of all possible clusters
        prps = compute_cluster_properties(
            site_species, interaction_matrix, self.bonds, self.n_bonds
        )
        self.cluster_energies = prps[0]
        self.cluster_site_species_occupancies = prps[1]
        self.mean_field_interactions = prps[2]

        m = sp.Matrix(self.cluster_site_species_occupancies)
        self.independent_cluster_occupancies = self.cluster_site_species_occupancies[
            independent_row_indices(m)
        ]
        self.clusters_as_independent_cluster_fractions = (
            np.linalg.lstsq(
                self.independent_cluster_occupancies.T.astype(float),
                self.cluster_site_species_occupancies.T.astype(float),
                rcond=0,
            )[0]
            .round(decimals=12)
            .T
        )

        self.mu_guess = np.zeros(len(self.independent_cluster_occupancies))


class DoubleEmbeddedCluster(Cluster):
    def __init__(self, site_species, single_cluster_bonds, n_bonds, interaction_matrix):
        self.name = "double"

        self.species_counts = [len(s) for s in site_species]
        self.total_site_species = int(sum(self.species_counts))
        self.total_sites = len(site_species)

        # flip the order of the clusters, must be int
        half_sites = int(self.total_sites / 2)
        bonds_2 = (single_cluster_bonds + half_sites) % self.total_sites
        self.bonds = np.concatenate((single_cluster_bonds, bonds_2), axis=0)

        n_bonds_2 = copy(n_bonds)
        self.n_bonds = np.concatenate((n_bonds, n_bonds_2), axis=0)

        # Calculates the energies of all possible clusters
        prps = compute_cluster_properties(
            site_species, interaction_matrix, self.bonds, self.n_bonds
        )
        self.cluster_energies = prps[0]
        self.full_cluster_site_species_occupancies = prps[1]
        full_mean_field_interactions = prps[2] * 2.0

        half_species = int(self.total_site_species / 2)
        self.cluster_site_species_occupancies = (
            prps[1][:, :half_species] + prps[1][:, half_species:]
        ) / 2.0

        m = sp.Matrix(self.cluster_site_species_occupancies)
        self.independent_cluster_occupancies = self.cluster_site_species_occupancies[
            independent_row_indices(m)
        ]
        self.clusters_as_independent_cluster_fractions = (
            np.linalg.lstsq(
                self.independent_cluster_occupancies.T.astype(float),
                self.cluster_site_species_occupancies.T.astype(float),
                rcond=0,
            )[0]
            .round(decimals=12)
            .T
        )
        self.mean_field_interactions = (
            full_mean_field_interactions[:, :half_species]
            + full_mean_field_interactions[:, half_species:]
        ) / 2.0
        self.mu_guess = np.zeros(len(self.independent_cluster_occupancies))


fig = plt.figure(figsize=(10, 4))
ax = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

# Number of species
n_species = 2

# Symmetric interaction matrix for 2 species
interaction_matrix = np.array(
    [
        [0, -1],
        [-1, 0],
    ]
)

# 1) Simple cluster as part of infinite matrix
# Allowed species on each site
site_species = [[0, 1]] * 4

# Bonds between sites, including half-bonds to other clusters
bonds = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

# Multiplicity of bonds of each type
n_bonds = np.array([2.0, 2.0, 2.0, 2.0])

model1 = SingleCluster(site_species, bonds, n_bonds, interaction_matrix)
model1.print_clusters()
model1.energy_histogram(ax[0])


# Allowed species on each site
site_species = [[0, 1]] * 8

# Bonds between sites, including half-bonds outside the current cluster
bonds = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 4],
        [4, 1],
        [5, 2],
        [6, 3],
        [7, 0],
    ]
)

# Multiplicity of bonds of each type
n_bonds = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

model2 = DoubleEmbeddedCluster(site_species, bonds, n_bonds, interaction_matrix)
model2.print_clusters()
model2.energy_histogram(ax[1])

ax[0].set_title("Single cluster")
ax[1].set_title("Embedded double cluster")

ax[0].set_xlim(-10, 2)
ax[1].set_xlim(-20, 4)

fig.set_layout_engine("tight")
plt.show()


mean_field_fractions = [0.0, 0.25, 0.5, 0.75, 1.0 - 1.0e-3]
# mean_field_fractions = [0.25]
fs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
fs = [0.5]

temperatures = np.linspace(0.4, 0.01, 10001)

fig = plt.figure(figsize=(10, 8))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

for b, (model, m, ls) in enumerate([(model1, 1.0, "--"), (model2, 2.0, "-")]):
    c = -1
    for f in fs:
        model.mu_guess = np.array([0.0] * 5)

        for d, mean_field_fraction in enumerate(mean_field_fractions):
            c += 1
            print(
                f"LRO fraction: {np.abs(2.*f - 1.)}, mean field fraction: {mean_field_fraction}"
            )
            site_fractions = np.array([f, 1.0 - f, 1.0 - f, f, f, 1.0 - f, 1.0 - f, f])

            reduced_independent_cluster_fractions = model.independent_cluster_fractions(
                site_fractions
            )

            entropies = np.empty_like(temperatures)
            energies = np.empty_like(temperatures)

            for i, T in enumerate(temperatures):
                mu_independent_clusters = model.mu(
                    T, reduced_independent_cluster_fractions, mean_field_fraction
                )

                E_clusters = effective_cluster_energies(
                    reduced_independent_cluster_fractions,
                    model.independent_cluster_occupancies,
                    model.cluster_energies,
                    model.mean_field_interactions,
                    mean_field_fraction,
                )

                p = cluster_probabilities(
                    mu_independent_clusters,
                    T,
                    E_clusters,
                    model.clusters_as_independent_cluster_fractions,
                )

                if np.abs(sum(p) - 1) > 1.0e-6:
                    print(
                        f"{sum(p)}, {model.name}, c={f}, mff={mean_field_fraction}, T={T}"
                    )
                    entropies[i] = np.nan
                    energies[i] = np.nan
                else:
                    entropies[i] = -R * np.sum(p * np.log(p))
                    energies[i] = p.dot(E_clusters)

            ax[0].plot(R * temperatures, entropies / m, c=colour[c], linestyle=ls)
            ax[1].plot(R * temperatures, energies / m, c=colour[c], linestyle=ls)
            ax[2].plot(
                R * temperatures,
                np.gradient(energies, entropies, edge_order=2) - temperatures,
                c=colour[c],
                linestyle=ls,
                label=f"{model.name}, c:{f}, f(mean):{mean_field_fraction}",
            )
            ax[3].plot(
                R * temperatures,
                (energies - temperatures * entropies) / m,
                c=colour[c],
                linestyle=ls,
            )


ax[0].set_ylim(
    0,
)

ax[0].set_ylabel("Entropies")
ax[1].set_ylabel("Energies")
ax[2].set_ylabel("dEdS - T")
ax[3].set_ylabel("Helmholtz")

ax[2].legend()

plt.show()
