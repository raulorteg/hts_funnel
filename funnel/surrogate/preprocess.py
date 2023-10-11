#!/usr/bin/env python3

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import spatial

from funnel.surrogate.definitions import ATOMICNUMBER_DICT


def load_dict(filename: str) -> dict:
    """
    Loads the preprocessing dictionary saved in the dataset folder so that
    processing can be done consistently

    :param str filename: string path to the file where the preprocessing information is saved.
    :returns: collections.defaultdict object with the loaded dictionary used in preprocessing.
    :rtype: collections.defaultdict
    """
    with open(filename, "rb") as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values()) + 1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default


def create_sphere(radius: float, grid_interval: float):
    """
    Create the sphere to be placed on each atom of a molecule.

    :param float radius: radius hyperparameter to be used in the sphere creation.
    :param float grid_interval: hyperparameter, distance between adjacent positions in the gridding space.
    :returns: numpy.array collection of x,y,z coordinates within the sphere.
    :rtype: numpy.array
    """
    xyz = np.arange(-radius, radius + 1e-3, grid_interval)
    sphere = [
        [x, y, z]
        for x in xyz
        for y in xyz
        for z in xyz
        if (x**2 + y**2 + z**2 <= radius**2) and [x, y, z] != [0, 0, 0]
    ]
    return np.array(sphere)


def create_field(sphere, coords):
    """Create the grid field of a molecule."""
    field = [f for coord in coords for f in sphere + coord]
    return np.array(field)


def create_orbitals(orbitals, orbital_dict):
    """
    Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p) into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2,
    where coords = [[x_1, y_1, z_1], [x_2, y_2, z_2], ...].
    For example, when coords1 is field_coords and coords2 is atomic_coords
    of a molecule, each element of the matrix is the distance
    between a field point and an atomic position in the molecule.
    Note that we transform all 0 elements in the distance matrix
    into a large value (e.g., 1e6) because we use the Gaussian:
    exp(-d^2), where d is the distance, and exp(-1e6^2) becomes 0.
    """
    distance_matrix = spatial.distance_matrix(coords1, coords2)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def create_potential(distance_matrix, atomic_numbers):
    """Create the Gaussian external potential used in Brockherde et al., 2017,
    Bypassing the Kohn-Sham equations with machine learning.
    """
    Gaussians = np.exp(-(distance_matrix**2))
    return -np.matmul(Gaussians, atomic_numbers)
