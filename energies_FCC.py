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
from scipy.optimize import minimize
from A3B_data import H_full_A3B, S_full_A3B, T_full_A3B
from skimage.restoration import inpaint_biharmonic
from skimage import img_as_float
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
from scipy.interpolate import RegularGridInterpolator


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


def fill_nans_inpaint(arr):
    from numpy import isnan

    arr = img_as_float(arr)
    mask = np.isnan(arr)
    arr[mask] = 0  # Dummy value; gets replaced
    filled = inpaint_biharmonic(arr, mask, channel_axis=None)
    return filled


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
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
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


def cluster_site_correlations(full_cluster_occupancies, site_species_counts):
    n_clusters, _ = full_cluster_occupancies.shape
    n_sites = len(site_species_counts)

    idx = 0
    site_correlations = np.empty((n_clusters, n_clusters, n_sites))
    for i, n_species_on_site_i in enumerate(site_species_counts):
        cl = full_cluster_occupancies[:, idx : idx + n_species_on_site_i]
        site_correlations[:, :, i] = cl @ cl.T
        idx += n_species_on_site_i

    return site_correlations


def mean_field_cluster_energies(model, site_occupancies):
    """
    Compute mean-field cluster proportions and energies using flattened 1-hot cluster encodings.

    Parameters
    ----------
    x_flat : np.ndarray
        Flattened array of mean site-species occupancies, shape (total_site_species,)

    c_mat : np.ndarray
        2D array of one-hot encoded clusters, shape (n_clusters, total_site_species)

    energies : np.ndarray
        Array of cluster energies, shape (n_clusters,)

    site_species_counts : list or np.ndarray
        Number of species per site, length = number of sites

    Returns
    -------
    p : np.ndarray
        Mean-field proportions for each cluster, shape (n_clusters,)

    E_mf : np.ndarray
        Mean-field energy contributions for each cluster, shape (n_clusters,)
    """

    # Step 1: Compute the probability q_ij
    # (q_ij = sum_k c_ijk * x_jk) that each site j
    # is occupied by the species found on that site in
    # cluster i.
    q = np.zeros((model.n_clusters, model.total_sites), dtype=float)
    idx = 0

    # Fudge for double site
    if type(model) == DoubleEmbeddedCluster:
        site_occupancies = np.concatenate((site_occupancies, site_occupancies))

    for j, n_species_on_site_j in enumerate(model.site_species_counts):
        # Extract the relevant slices for site j
        xj = site_occupancies[idx : idx + n_species_on_site_j]
        cj = model.full_cluster_site_species_occupancies[
            :, idx : idx + n_species_on_site_j
        ]
        q[:, j] = cj @ xj
        idx += n_species_on_site_j

    # Step 2: For each site l, compute r_il = prod_{j \neq l} q_ij,
    # the probability that the species in all the sites
    # surrounding site l (but not site l itself) match the
    # occupancies of cluster i
    r = np.ones((model.n_clusters, model.total_sites), dtype=float)
    for i_site in range(model.total_sites):
        # Exclude column l from product
        q_mod = q.copy()
        q_mod[:, i_site] = 1.0
        r[:, i_site] = np.prod(q_mod, axis=1)

    # Step 3: Compute the proportions of clusters i
    # associated with each cluster m
    # Equivalent einsum indices are (mil, il -> mi)
    p = np.sum(model.cluster_site_correlations * r, axis=2) / model.total_sites

    # Step 4: Compute effective cluster energies E_m
    # given the cluster proportions p_mi
    return p @ model.cluster_energies


def mean_field_cluster_energies_simple(model, site_occupancies):
    return model.mean_field_interactions.dot(site_occupancies)


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
    model,
    independent_cluster_fractions,
    mean_field_fraction,
):
    ind = model.independent_cluster_occupancies
    site_species_occupancies = ind.T.dot(independent_cluster_fractions)

    # TODO: Choose between mean_field_cluster_energies_simple and mean_field_cluster_energies
    mean_field_energies = mean_field_cluster_energies_simple(
        model, site_species_occupancies
    )

    f = mean_field_fraction
    return (1.0 - f) * model.cluster_energies + f * mean_field_energies


def delta_independent_cluster_proportions(
    mu_independent_clusters,
    model,
    T,
    independent_cluster_fractions,
    mean_field_fraction,
    clusters_as_independent_cluster_fractions,
):
    E_clusters = effective_cluster_energies(
        model,
        independent_cluster_fractions,
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
    model,
    T,
    independent_cluster_fractions,
    mean_field_fraction,
    clusters_as_independent_cluster_fractions,
    guess=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
):
    sol = root(
        delta_independent_cluster_proportions,
        guess,
        args=(
            model,
            T,
            independent_cluster_fractions,
            mean_field_fraction,
            clusters_as_independent_cluster_fractions,
        ),
        method="lm",
    )

    if not sol.success:
        sol = root(
            delta_independent_cluster_proportions,
            guess,
            args=(
                model,
                T,
                independent_cluster_fractions,
                mean_field_fraction,
                clusters_as_independent_cluster_fractions,
            ),
            method="hybr",
        )

    # assert sol.success
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
            self,
            temperature,
            independent_cluster_fractions,
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
        self.site_species_counts = [len(s) for s in site_species]
        self.total_site_species = int(sum(self.site_species_counts))
        self.total_sites = len(site_species)

        self.bonds = bonds
        self.n_bonds = n_bonds

        # Calculates the energies of all possible clusters
        prps = compute_cluster_properties(
            site_species, interaction_matrix, self.bonds, self.n_bonds
        )
        self.cluster_energies = prps[0]
        self.n_clusters = len(prps[0])
        self.full_cluster_site_species_occupancies = prps[1]
        self.cluster_site_species_occupancies = prps[1]
        self.mean_field_interactions = prps[2]
        self.cluster_site_correlations = cluster_site_correlations(
            self.full_cluster_site_species_occupancies, self.site_species_counts
        )

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

        self.site_species_counts = [len(s) for s in site_species]
        self.total_site_species = int(sum(self.site_species_counts))
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
        self.n_clusters = len(prps[0])
        self.full_cluster_site_species_occupancies = prps[1]
        full_mean_field_interactions = prps[2] * 2.0
        self.cluster_site_correlations = cluster_site_correlations(
            self.full_cluster_site_species_occupancies, self.site_species_counts
        )

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


class CSA(Cluster):
    """
    From Oates et al., 1999
    """

    def __init__(self, WAB, alpha, beta):

        self.name = "FCC CSA"

        site_species = [[0, 1]] * 4
        self.site_species_counts = [len(s) for s in site_species]
        self.total_site_species = int(sum(self.site_species_counts))
        self.total_sites = len(site_species)

        self.full_cluster_site_species_occupancies = (
            compute_cluster_site_species_occupancies(site_species)
        )
        self.cluster_site_species_occupancies = (
            compute_cluster_site_species_occupancies(site_species)
        )
        self.cluster_site_correlations = cluster_site_correlations(
            self.full_cluster_site_species_occupancies, self.site_species_counts
        )

        nA = np.sum(self.cluster_site_species_occupancies[:, ::2], axis=1)
        E = np.zeros_like(nA, dtype=float)
        E[nA == 1] = 3.0 * WAB * (1.0 + alpha)
        E[nA == 2] = WAB * 4.0
        E[nA == 3] = 3.0 * WAB * (1.0 + beta)

        self.cluster_energies = E
        self.n_clusters = len(E)
        self.mean_field_interactions = np.zeros((16, 8))

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

        self.mu_guess = np.zeros(len(self.independent_cluster_occupancies)) - 8.0


def site_fractions_from_cf(c, f):
    cT = c * 4.0
    if cT >= 3.0:
        cR = cT - 3.0
        site_fractions_o = np.array([1.0 - cR, cR, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    elif cT >= 2.0:
        cR = cT - 2.0
        site_fractions_o = np.array([1.0 - cR, cR, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    elif cT >= 1.0:
        cR = cT - 1.0
        site_fractions_o = np.array([1.0 - cR, cR, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    else:
        site_fractions_o = np.array([1.0 - cT, cT, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    site_fractions_d = np.array([1.0 - c, c] * 4)

    site_fractions = (1.0 - f) * site_fractions_o + f * site_fractions_d
    return site_fractions


def helmholtz_at_OCT(farr, c, T, mean_field_fraction, model):
    site_fractions = site_fractions_from_cf(c, farr[0])

    reduced_independent_cluster_fractions = model.independent_cluster_fractions(
        site_fractions
    )
    mu_independent_clusters = model.mu(
        T, reduced_independent_cluster_fractions, mean_field_fraction
    )

    E_clusters = effective_cluster_energies(
        model,
        reduced_independent_cluster_fractions,
        mean_field_fraction,
    )

    p = cluster_probabilities(
        mu_independent_clusters,
        T,
        E_clusters,
        model.clusters_as_independent_cluster_fractions,
    )

    entropy = -R * np.sum(p * np.log(p))
    energy = p.dot(E_clusters)

    return energy - T * entropy


def energy_entropy_order_equilibrated_at_CT(
    c, T, mean_field_fraction, model, f_guess=0.001
):
    sol = minimize(
        helmholtz_at_OCT,
        [f_guess],
        args=(c, T, mean_field_fraction, model),
        bounds=[(0.0001, 1.0)],
        tol=1.0e-5,
        method="Nelder-Mead",
        options={"maxiter": 500},
    )
    if not sol.success and sol.message != 2:
        sol = minimize(
            helmholtz_at_OCT,
            [f_guess],
            args=(c, T, mean_field_fraction, model),
            bounds=[(0.0001, 1.0)],
            tol=1.0e-5,
            options={"maxiter": 500},
        )

    site_fractions = site_fractions_from_cf(c, sol.x[0])

    reduced_independent_cluster_fractions = model.independent_cluster_fractions(
        site_fractions
    )
    mu_independent_clusters = model.mu(
        T, reduced_independent_cluster_fractions, mean_field_fraction
    )

    E_clusters = effective_cluster_energies(
        model,
        reduced_independent_cluster_fractions,
        mean_field_fraction,
    )

    p = cluster_probabilities(
        mu_independent_clusters,
        T,
        E_clusters,
        model.clusters_as_independent_cluster_fractions,
    )

    entropy = -R * np.sum(p * np.log(p))
    energy = p.dot(E_clusters)

    return energy, entropy, sol.x[0], sol, p


if __name__ == "__main__":
    if True:
        temperatures = np.linspace(1.8, 0.02, 100)
        energies = np.zeros_like(temperatures)
        entropies = np.zeros_like(temperatures)
        cluster_entropies = np.zeros_like(temperatures)

        # Load the image
        img1 = mpimg.imread("figures/Oates_et_al_1999_Figure_2.png")
        img2 = mpimg.imread("figures/Ferreira_et_al_2018_Figure_1a.png")
        img3 = mpimg.imread("figures/Jindal_et_al_2014_Figure_5.png")
        img4 = mpimg.imread("figures/Jindal_et_al_2014_Figure_6.png")

        # Define the plot
        fig = plt.figure(figsize=(10, 8))
        ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

        ax[0].fill_between(
            [0.0, 1.8],
            [np.log(6) / 4.0, np.log(6) / 4.0],
            [np.log(16) / 4.0, np.log(16) / 4.0],
            alpha=0.1,
            color="red",
            label="range of cluster entropies",
        )

        # Show the image with specified bounds
        ax[0].imshow(img1, extent=[0.6, 1.2, 0.0, 0.6], aspect="auto")
        ax[1].imshow(img2, extent=[0.6, 1.4, -4.25, -3], aspect="auto")
        ax[2].imshow(img2, extent=[0.6, 1.4, -4.25, -3], aspect="auto")

        mean_field_fraction = 0.0
        n = 4.0

        for gamma in [1.0, 1.22]:

            WAB = -R / gamma
            alpha = beta = 0.0
            model = CSA(WAB, alpha, beta)

            for i, T in enumerate(temperatures):

                f = 0.5
                site_fractions = np.array([f, 1.0 - f, 1.0 - f, f, f, 1.0 - f, 1.0 - f, f])

                independent_cluster_fractions = model.independent_cluster_fractions(
                    site_fractions
                )
                mu_independent_clusters = model.mu(
                    T, independent_cluster_fractions, mean_field_fraction
                )

                E_clusters = effective_cluster_energies(
                    model,
                    independent_cluster_fractions,
                    mean_field_fraction,
                )

                p = cluster_probabilities(
                    mu_independent_clusters,
                    T,
                    E_clusters,
                    model.clusters_as_independent_cluster_fractions,
                )

                point_entropy = -R * np.sum(site_fractions * np.log(site_fractions))
                cluster_entropy = -R * np.sum(p * np.log(p))

                entropies[i] = (
                    float(gamma) * cluster_entropy
                    - (float(gamma) - 1.0 / float(n)) * point_entropy
                )
                cluster_entropies[i] = cluster_entropy / float(n)

                energies[i] = gamma * np.sum(p.dot(E_clusters))

            mask = temperatures > 0.9
            ln = ax[0].plot(temperatures, entropies / R, linestyle=":")
            ax[0].plot(
                temperatures[mask],
                entropies[mask] / R,
                c=ln[0].get_color(),
                label=f"total, $\\gamma = {{{gamma}}}$",
            )
            ax[0].plot(
                temperatures,
                cluster_entropies / R,
                c=ln[0].get_color(),
                linestyle="--",
                label=f"cluster-only, $\\gamma = {{{gamma}}}$",
            )
            ax[0].plot(
                temperatures,
                np.log(16)
                / (np.log(16.0) - np.log(6))
                * (cluster_entropies / R - np.log(6) / 4.0),
                c=ln[0].get_color(),
                linestyle=":",
                label=f"scaled cluster-only, $\\gamma = {{{gamma}}}$",
            )

            helmholtz = energies - temperatures * entropies
            ax[1].plot(temperatures, energies / R, c=ln[0].get_color())
            ax[2].plot(temperatures, helmholtz / R, c=ln[0].get_color())

            ax[2].plot(temperatures, -3.0 - temperatures * np.log(2), linestyle=":")

        custom_handles = [
            Line2D(
                [],
                [],
                marker="d",
                color="black",
                linestyle="None",
                markersize=8,
                label="MC",
            ),
            Line2D(
                [],
                [],
                marker="^",
                markerfacecolor="none",
                color="black",
                linestyle="None",
                markersize=8,
                label="CVM",
            ),
            Line2D(
                [],
                [],
                marker="x",
                color="black",
                linestyle="None",
                markersize=8,
                label="CSA, $\\gamma = 1$",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="None",
                markersize=8,
                label="CSA, $\\gamma = 1.22$",
            ),
        ]

        # Add the legend
        handles, labels = ax[0].get_legend_handles_labels()
        all_handles = custom_handles + handles
        all_labels = [h.get_label() for h in custom_handles] + labels

        ax[0].legend(all_handles, all_labels, frameon=True, fontsize=8)

        ax[0].set_xlim(
            0.0,
        )
        ax[0].set_ylim(0.0, 0.7)
        plt.show()

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
        bonds = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

        # Multiplicity of bonds of each type
        n_bonds = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        model1 = SingleCluster(site_species, bonds, n_bonds, interaction_matrix)
        model1.print_clusters()
        model1.energy_histogram(ax[0])

        # Allowed species on each site
        site_species = [[0, 1]] * 8

        # Bonds between sites, including half-bonds outside the current cluster
        bonds = np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                [2, 3],
                [0, 5],
                [0, 6],
                [0, 7],
                [1, 4],
                [1, 6],
                [1, 7],
                [2, 4],
                [2, 5],
                [2, 7],
                [3, 4],
                [3, 5],
                [3, 6],
            ]
        )

        # Multiplicity of bonds of each type
        n_bonds = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )

        model2 = DoubleEmbeddedCluster(site_species, bonds, n_bonds, interaction_matrix)
        model2.print_clusters()
        model2.energy_histogram(ax[1])

        ax[0].set_title("Single cluster")
        ax[1].set_title("Embedded double cluster")

        ax[0].set_xlim(-10, 2)
        ax[1].set_xlim(-20, 4)

        fig.set_layout_engine("tight")
        plt.show()


    if True:
        mff = 0.22  # 0.22 works well for AB
        # mean_field_fractions = [0., 0.25, 0.5, 0.75, 1.-1.e-3]
        mean_field_fractions = [mff]
        fs = np.linspace(1.0e-6, 0.5, 6)
        models = [model1, model2]
        models = [model2]
        #  fs = [0.5]

        temperatures = np.linspace(0.10, 0.01, 101)

        fig = plt.figure(figsize=(10, 8))
        ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

        ax[0].imshow(img1, extent=[0.6 / 2.0, 1.2 / 2.0, 0.0, 0.6 * 4.0 * R], aspect="auto")
        ax[1].imshow(img3, extent=[1.5 / 4.0, 2.5 / 4.0, -8.0, -7.0], aspect="auto")
        ax[3].imshow(
            img2, extent=[0.6 / 2.0, 1.4 / 2.0, -4.25 * 2.0, -3 * 2.0], aspect="auto"
        )

        for b, model in enumerate(models):

            m = 1.0
            ls = "--"

            if model == model2:
                m = 2.0
                ls = "-"

            c = -1
            for f in fs:
                model.mu_guess = np.array([0.0] * 5)

                for d, mean_field_fraction in enumerate(mean_field_fractions):
                    print(d)
                    #  mean_field_fraction = 0.22 * 4 * f * (1. - f)
                    c += 1
                    print(
                        f"LRO fraction: {np.abs(2.*f - 1.)}, mean field fraction: {mean_field_fraction}"
                    )
                    site_fractions = np.array(
                        [f, 1.0 - f, 1.0 - f, f, f, 1.0 - f, 1.0 - f, f]
                    )

                    reduced_independent_cluster_fractions = (
                        model.independent_cluster_fractions(site_fractions)
                    )

                    entropies = np.empty_like(temperatures)
                    energies = np.empty_like(temperatures)

                    for i, T in enumerate(temperatures):
                        print(i, T)
                        mu_independent_clusters = model.mu(
                            T, reduced_independent_cluster_fractions, mean_field_fraction
                        )

                        E_clusters = effective_cluster_energies(
                            model,
                            reduced_independent_cluster_fractions,
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

            # Calculate equilibrated properties
            temperatures = np.linspace(0.1, 0.01, 101)
            entropies = np.empty_like(temperatures)
            energies = np.empty_like(temperatures)
            orders = np.empty_like(temperatures)
            p_clusters = np.empty((len(temperatures), 256))
            for d, mean_field_fraction in enumerate(mean_field_fractions):
                model.mu_guess = np.array([0.0] * 5)
                for i, T in enumerate(temperatures):
                    print(i, T)
                    sol = energy_entropy_order_equilibrated_at_CT(
                        0.5, T, mean_field_fraction, model
                    )
                    if sol[3].success or sol[3].status == 2:
                        energies[i] = sol[0]
                        entropies[i] = sol[1]
                        orders[i] = sol[2]
                        p_clusters[i] = sol[4]
                    else:
                        energies[i] = np.nan
                        entropies[i] = np.nan
                        orders[i] = np.nan
                        p_clusters[i] = np.nan

                ax[0].plot(R * temperatures, entropies / m, c="black")
                ax[1].plot(R * temperatures, energies / m, c="black")
                ax[3].plot(R * temperatures, energies / m, c="black")
                ax[3].plot(
                    R * temperatures, (energies - temperatures * entropies) / m, c="black"
                )
                ax[3].plot(R * temperatures, -6.0 - temperatures * entropies / m, c="black")

        ax[0].set_ylim(
            0,
        )

        custom_handles = [
            Line2D(
                [],
                [],
                marker="d",
                color="black",
                linestyle="None",
                markersize=8,
                label="MC",
            ),
            Line2D(
                [],
                [],
                marker="^",
                markerfacecolor="none",
                color="black",
                linestyle="None",
                markersize=8,
                label="CVM",
            ),
            Line2D(
                [],
                [],
                marker="x",
                color="black",
                linestyle="None",
                markersize=8,
                label="CSA, $\\gamma = 1$",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                linestyle="None",
                markersize=8,
                label="CSA, $\\gamma = 1.22$",
            ),
        ]

        # Add the legend
        handles, labels = ax[0].get_legend_handles_labels()
        all_handles = custom_handles + handles
        all_labels = [h.get_label() for h in custom_handles] + labels

        ax[0].legend(all_handles, all_labels, frameon=True, fontsize=8)

        ax[0].set_ylabel("Entropies")
        ax[1].set_ylabel("Energies")
        ax[2].set_ylabel("dEdS - T")
        ax[3].set_ylabel("Helmholtz")

        ax[2].legend()

        plt.show()

    if True:
        # Disordered at 0.75
        mff = 0.25  # 0.25 works well for A3B
        temperatures = np.linspace(0.1, 0.01, 101)

        mean_field_fraction = mff
        model = model2
        m = 2.0

        f = 0.75
        site_fractions_disordered = np.array(
            [f, 1.0 - f, f, 1.0 - f, f, 1.0 - f, f, 1.0 - f]
        )
        site_fractions_ordered = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

        fig = plt.figure(figsize=(10, 8))
        ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

        ax[1].imshow(img4, extent=[1.5 / 4.0, 2.5 / 4.0, -6.0, -5.0], aspect="auto")

        fs = np.linspace(1.0, 1.0e-6, 6)
        for f in fs:
            model.mu_guess = np.zeros(5) - 10.0
            site_fractions = site_fractions_disordered * f + site_fractions_ordered * (
                1.0 - f
            )

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
                    model,
                    reduced_independent_cluster_fractions,
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

            ax[0].plot(R * temperatures, (entropies) / m)
            ax[1].plot(R * temperatures, energies / m)
            ax[2].plot(R * temperatures, (energies - temperatures * (entropies)) / m)

        # Calculate equilibrated properties
        temperatures = np.linspace(0.1, 0.01, 201)
        entropies = np.empty_like(temperatures)
        energies = np.empty_like(temperatures)
        orders = np.empty_like(temperatures)
        p_clusters = np.empty((len(temperatures), 256))

        model.mu_guess = np.array([0.0] * 5)
        for i, T in enumerate(temperatures):
            print(i, T)
            sol = energy_entropy_order_equilibrated_at_CT(
                0.25, T, mean_field_fraction, model
            )
            if sol[3].success or sol[3].status == 2:
                energies[i] = sol[0]
                entropies[i] = sol[1]
                orders[i] = sol[2]
                p_clusters[i] = sol[4]
            else:
                energies[i] = np.nan
                entropies[i] = np.nan
                orders[i] = np.nan
                p_clusters[i] = np.nan
        ax[0].plot(R * temperatures, entropies / m, c="black")
        ax[1].plot(R * temperatures, energies / m, c="black")
        ax[2].plot(R * temperatures, (energies - temperatures * entropies) / m, c="black")

        ax[0].plot(R * T_full_A3B, S_full_A3B, c="black", linestyle="--", linewidth=0.5)
        ax[1].plot(R * T_full_A3B, H_full_A3B, c="black", linestyle="--", linewidth=0.5)
        ax[2].plot(
            R * T_full_A3B,
            H_full_A3B - T_full_A3B * S_full_A3B,
            c="black",
            linestyle="--",
            linewidth=0.5,
        )

        ax[0].plot(
            R * temperatures,
            temperatures * 0.0 - 4.0 * R * (0.75 * np.log(0.75) + 0.25 * np.log(0.25)),
        )
        ax[1].plot(R * temperatures, temperatures * 0.0 - 4.5)
        plt.show()


    n_c = 199
    n_T = 199

    if True:
        compositions = np.linspace(0.5, 0.005, n_c)
        temperatures = np.linspace(0.1, 0.001, n_T)
        mean_field_fraction = 0.25

        energies = np.zeros((len(temperatures), len(compositions)))
        entropies = np.zeros((len(temperatures), len(compositions)))
        orders = np.zeros((len(temperatures), len(compositions)))
        p_clusters = np.empty((len(temperatures), len(compositions), 256))

        m = 2.0

        for j, c in enumerate(compositions):
            print(f"c: {c}")
            E_max = -32.0 * c
            model2.mu_guess = np.array([E_max] * 5)
            for i, T in enumerate(temperatures):
                mu_guess = model2.mu_guess
                sol = energy_entropy_order_equilibrated_at_CT(
                    c, T, mean_field_fraction, model2
                )

                if sol[3].success is False:
                    model2.mu_guess = np.array([-16] * 5)
                    sol = energy_entropy_order_equilibrated_at_CT(
                        c, T, mean_field_fraction, model2
                    )

                if sol[3].success or sol[3].status == 2:
                    energies[i, j] = sol[0]
                    entropies[i, j] = sol[1]
                    orders[i, j] = sol[2]
                    p_clusters[i, j] = sol[4]
                else:
                    print(sol)
                    energies[i, j] = np.nan
                    entropies[i, j] = np.nan
                    orders[i, j] = np.nan
                    p_clusters[i, j] = np.nan
                    model2.mu_guess = mu_guess

                if entropies[i, j] == np.nan:
                    print(T, energies[i, j], entropies[i, j], orders[i, j])

        cc, tt = np.meshgrid(compositions, temperatures)

        cc = cc.flatten()
        tt = tt.flatten()
        ee = energies.flatten()
        ss = entropies.flatten()
        oo = orders.flatten()
        np.savetxt("output/XTESO.dat", np.array([cc, tt, ee, ss, oo]).T)

    if True:
        cc, tt, ee, ss, oo = np.loadtxt("output/XTESO.dat", unpack=True)

        # note need to fix when rerun
        Tclip = -12
        cc = cc.reshape(n_T, n_c)[:Tclip]
        tt = tt.reshape(n_T, n_c)[:Tclip]

        compositions = cc[0, :]
        temperatures = tt[:, 0]

        energies = ee.reshape(n_T, n_c)[:Tclip]
        entropies = ss.reshape(n_T, n_c)[:Tclip]
        orders = oo.reshape(n_T, n_c)[:Tclip]

        entropies[(orders < 0.001)] = np.nan
        energies[(orders < 0.01)] = np.nan
        orders[(orders < 0.01)] = np.nan

        mean_field_fraction = 0.25

        model2.mu_guess = np.array([0.0, -12.0, -12.0, -12.0, -12.0])
        for (j, i), entropy in np.ndenumerate(entropies.T):
            if np.isnan(entropy):
                print(i, j)
                F = helmholtz_at_OCT(
                    [orders[i, j]], cc[i, j], tt[i, j], mean_field_fraction, model2
                )
                S = -(F - energies[i, j]) / tt[i, j]
                E, S, O, sol, p = energy_entropy_order_equilibrated_at_CT(
                    cc[i, j], tt[i, j], mean_field_fraction, model2
                )

                if S < 50.0 and S > 0:
                    print(model2.mu_guess)
                    energies[i, j] = E
                    entropies[i, j] = S
                    orders[i, j] = O
                    p_clusters[i, j] = p

                else:
                    model2.mu_guess = np.array([0.0, -12.0, -12.0, -12.0, -12.0])
        for (i, j), entropy in np.ndenumerate(entropies):
            if np.isnan(entropy):
                print(i, j)
                F = helmholtz_at_OCT(
                    [orders[i, j]], cc[i, j], tt[i, j], mean_field_fraction, model2
                )
                S = -(F - energies[i, j]) / tt[i, j]
                E, S, _, sol, p = energy_entropy_order_equilibrated_at_CT(
                    cc[i, j], tt[i, j], mean_field_fraction, model2
                )

                if S < 50.0 and S > 0:
                    print(model2.mu_guess)
                    energies[i, j] = E
                    entropies[i, j] = S
                    orders[i, j] = O
                    p_clusters[i, j] = p
                else:
                    model2.mu_guess = np.array([0.0, -12.0, -12.0, -12.0, -12.0])

        # Only infill as a last resort
        entropies = fill_nans_inpaint(entropies)
        energies = fill_nans_inpaint(energies)
        orders = fill_nans_inpaint(orders)

        cc = cc.flatten()
        tt = tt.flatten()
        ee = energies.flatten()
        ss = entropies.flatten()
        oo = orders.flatten()
        np.savetxt("output/XTESO_new.dat", np.array([cc, tt, ee, ss, oo]).T)

    cc, tt, ee, ss, oo = np.loadtxt("output/XTESO_new.dat", unpack=True)

    n_T = n_T - 12
    cc = cc.reshape(n_T, n_c)
    tt = tt.reshape(n_T, n_c)

    compositions = cc[0, :]
    temperatures = tt[:, 0]

    energies = ee.reshape(n_T, n_c)
    entropies = ss.reshape(n_T, n_c)
    orders = oo.reshape(n_T, n_c)

    helmholtz = energies - tt * entropies

    m = 2.0
    mean_field_fraction = 0.25

    if False:
        # Interpolation (this method does not work)
        interpz = RegularGridInterpolator(
            (temperatures, compositions), entropies, method="linear"
        )
        interpc = RegularGridInterpolator(
            (temperatures, compositions), orders, method="linear"
        )

        xnew = np.linspace(compositions.min(), compositions.max(), 100)
        ynew = np.linspace(temperatures.min(), temperatures.max(), 100)
        xxnew, yynew = np.meshgrid(xnew, ynew)
        points = np.stack((yynew.ravel(), xxnew.ravel()), axis=-1)
        zznew = interpz(points).reshape(100, 100)
        ccnew = interpc(points).reshape(100, 100)
    else:
        zznew = entropies[:, ::-1]
        xxnew = tt[:, ::-1]
        yynew = cc[:, ::-1]
        ccnew = orders[:, ::-1]

    xxnew = np.concatenate((xxnew[:, :-1], xxnew[:, ::-1]), axis=1)
    yynew = np.concatenate((yynew[:, :-1], 1. - yynew[:, ::-1]), axis=1)
    zznew = np.concatenate((zznew[:, :-1], zznew[:, ::-1]), axis=1)
    ccnew = np.concatenate((ccnew[:, :-1], ccnew[:, ::-1]), axis=1)

    # 3D figure
    fig = go.Figure(
        data=[
            go.Surface(
                z=zznew,
                x=xxnew,
                y=yynew,
                surfacecolor=ccnew,
                colorscale="turbo",
                colorbar=dict(title="disorder"),
            )
        ]
    )

    fig.update_layout(
        width=1400,  # Twice as wide
        height=1000,  # Standard height
        scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Composition',
            zaxis_title='Entropy',
            aspectratio=dict(x=1, y=2, z=1)
        )
    )

    fig.show()


    # 2D figures
    fig = plt.figure(figsize=(10, 8))
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]


    cmap = cm.rainbow

    labels = ["long range order", "Entropy/R", "Energy", "Helmholtz"]
    properties = [1.0 - orders, entropies / R, energies, helmholtz]
    for i, c in enumerate(properties):
        norm = mcolors.Normalize(vmin=c.min(), vmax=c.max())
        surf = ax[i].contourf(cc, 4.0 * R * tt, c, cmap=cmap, levels=51)
        surf = ax[i].contourf(1.0 - cc, 4.0 * R * tt, c, cmap=cmap, levels=51)
        cbar = fig.colorbar(surf, ax=ax[i], label=labels[i])

        if i == 0:
            levels = [0.0000001, 0.3, 0.6, 0.8, 0.9, 0.95]
            for level in levels:
                contour = ax[i].contour(cc, tt, c, levels=[level])

                # Extract contour paths
                contour_paths = []
                for collection in contour.collections:
                    for path in collection.get_paths():
                        coords = path.vertices  # Nx2 array of (x, y)
                        contour_paths.append(coords)

                coords = np.vstack(contour_paths)

                for collection in contour.collections:
                    collection.remove()

                coords_reflected = coords.copy()
                coords_reflected[:, 0] = 1.0 - coords[:, 0]
                coords_reflected = coords_reflected[::-1]
                coords = np.vstack([coords, coords_reflected])

                # Use only unique coords
                _, unique_indices = np.unique(coords, axis=0, return_index=True)
                x = coords[unique_indices, 0]
                y = coords[unique_indices, 1]

                # Fit parametric spline (s is smoothing factor)
                tck, u = splprep([x, y], s=0.00002)

                # Evaluate the spline at a finer set of points
                u_fine = np.linspace(0, 1, 500)
                x_smooth, T_smooth = splev(u_fine, tck)

                # Plot smoothed contour on all plots
                for j in range(4):
                    ax[j].plot(
                        x_smooth,
                        4.0 * R * T_smooth,
                        c="black",
                        linewidth=1.0,
                        linestyle="-",
                    )

            ticks = np.linspace(0.0, 1.0, 6)
            new_labels = [f"{tick:.2f}" for tick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(new_labels)
            cbar.ax.hlines(
                levels,
                xmin=0,
                xmax=1,
                colors="black",
                linewidth=1,
                transform=cbar.ax.transAxes,
            )

        for j in range(4):
            ax[j].set_ylim(4.0 * R * temperatures.min(), 4.0 * R * temperatures.max())
            ax[j].set_xlim(0.0, 1.0)

    if True:
        img = mpimg.imread("figures/Binder_et_al_1981_Figure_5.png")

        ax[1].imshow(
            make_transparent(img), extent=[0.0, 1.0, 0, 2], aspect="auto", zorder=10
        )

    img = mpimg.imread("figures/Ackermann_et_al_1986_Figure_25.png")

    ax[1].imshow(make_transparent(img), extent=[0.8, 0.49, 0, 2], aspect="auto", zorder=20)


    img = mpimg.imread("figures/Inden_2001_Figure_8-15.png")
    ax[1].imshow(
        make_transparent(img), extent=[0.1, 0.9, 0, 2.56], aspect="auto", zorder=30
    )


    fig.set_layout_engine("tight")
    plt.show()
