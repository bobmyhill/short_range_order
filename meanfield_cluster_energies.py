import numpy as np
from itertools import product


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
    for l, n_species_on_site_l in enumerate(site_species_counts):
        cl = full_cluster_occupancies[
            :, idx : idx + n_species_on_site_l
        ]  # (n_clusters, Ml)
        site_correlations[:, :, l] = (
            cl @ cl.T
        )  # s_iml: (n_clusters, n_clusters), values in {0,1}
        idx += n_species_on_site_l

    return site_correlations


def mean_field_cluster_energies_flat(
    site_occupancies,
    full_cluster_occupancies,
    site_correlations,
    energies,
    site_species_counts,
):
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
    n_clusters, _ = full_cluster_occupancies.shape
    n_sites = len(site_species_counts)

    # Step 1: Compute the probability q_ij
    # (q_ij = sum_k c_ijk * x_jk) that each site j
    # is occupied by the species found on that site in
    # cluster i.
    q = np.zeros((n_clusters, n_sites), dtype=float)
    idx = 0
    for j, n_species_on_site_j in enumerate(site_species_counts):
        # Extract the relevant slices for site j
        xj = site_occupancies[idx : idx + n_species_on_site_j]
        cj = full_cluster_occupancies[:, idx : idx + n_species_on_site_j]
        q[:, j] = cj @ xj
        idx += n_species_on_site_j

    # Step 2: For each site l, compute r_il = prod_{j \neq l} q_ij,
    # the probability that the species in all the sites
    # surrounding site l (but not site l itself) match the
    # occupancies of cluster i
    r = np.ones((n_clusters, n_sites), dtype=float)
    for i_site in range(n_sites):
        # Exclude column l from product
        q_mod = q.copy()
        q_mod[:, i_site] = 1.0
        r[:, i_site] = np.prod(q_mod, axis=1)

    # Step 3: Compute the proportions of clusters i
    # associated with each cluster m
    # Equivalent einsum indices are (mil, il -> mi)
    p = np.sum(site_correlations * r, axis=2) / n_sites

    # Step 4: Compute effective cluster energies E_m
    # given the cluster proportions p_mi
    return p @ energies


if __name__ == "__main__":

    site_species = [[0, 1], [0, 1], [0, 1], [0, 1]]
    n_species_on_site = [len(site) for site in site_species]
    cluster_site_species_occupancies = compute_cluster_site_species_occupancies(
        site_species
    )
    site_correlations = cluster_site_correlations(
        cluster_site_species_occupancies, n_species_on_site
    )

    site_occupancies = np.array([0.4, 0.6, 0.2, 0.8, 0.7, 0.3, 0.7, 0.3])
    site_occupancies = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    site_occupancies = np.array([0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3])
    site_occupancies = np.array([1.0, 0.0] * 4)
    site_occupancies = np.array([0.0, 1.0] * 4)
    site_occupancies = np.array([0.5] * 8)

    n_A = np.sum(cluster_site_species_occupancies[:, ::2], axis=1)

    cluster_energies = np.zeros_like(n_A)

    cluster_energies[n_A == 2] = -8.0
    cluster_energies[n_A == 1] = -6.0
    cluster_energies[n_A == 3] = -6.0

    print(".......... cluster energies:", cluster_energies)

    E_mf = mean_field_cluster_energies_flat(
        site_occupancies,
        cluster_site_species_occupancies,
        site_correlations,
        cluster_energies,
        n_species_on_site,
    )

    print("Mean-field cluster energies:", E_mf)
