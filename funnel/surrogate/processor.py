import subprocess

import numpy as np
import torch

from funnel.surrogate.definitions import ATOMICNUMBER_DICT
from funnel.surrogate.hyperparameters import BASIS_SET, GRID_INTERVAL, RADIUS
from funnel.surrogate.preprocess import *


def smiles_to_xyz(in_filename: str, out_filename: str):
    """
    Uses OPENBABEL --gen3D command to convert SMILES strings into XYZ coordinates, minimizing the structure using UFF.
    :param str in_filename: Filepath to the SMILES file, containing a single column with SMILES strings for the molecules
    :param str out_filename: Filepath where the .xyz file coordinates are to be saved.
    """
    cmd_string = (
        f"obabel -ismi {in_filename} --gen3D -h --minimize --ff UFF -O {out_filename}"
    )
    subprocess.check_call(cmd_string, shell=True)

    with open(out_filename, "r") as f:
        chunks = f.read().strip().split("\n\n")[1:]

    with open(in_filename, "r") as f:
        smiles = f.read().strip().split("\n")

    if len(smiles) != len(chunks):
        raise Exception("Error: Some molecules could not be converted into xyz.")

    with open(out_filename, "w") as f:
        for smile, chunk in zip(smiles, chunks):
            chunk = "\n".join(chunk.split("\n")[:-1])
            print(smile, file=f)
            print(chunk + "\n", file=f)


class QDFprocessor(torch.utils.data.Dataset):
    def __init__(self, orbital_file: str):
        """
        Processor object used for converting the xyz coordinates of the molecules into the lists of quantum numbers, orbitals, etc to be fed to QuantumDeepField
        :param str orbital_file: String path to the file containing the orbital dictionary produced during the preprocessing step, when the dictionary is created.
        """

        # load the existing orbital dictionary from pickle file
        self.orbital_dict = load_dict(filename=orbital_file)

        # extract basis set
        inner_outer = [int(b) for b in BASIS_SET[:-1].replace("-", "")]
        self.basis_set_inner = inner_outer[0]
        self.basis_set_outer = sum(inner_outer[1:])

        # create a sphere for the gird field of a molecule
        self.sphere = create_sphere(radius=RADIUS, grid_interval=GRID_INTERVAL)

        # flag to prevent using the dataset without having used
        # the process_from_file() method that processes it
        self.empty = True

        # initialize empty list of molecular objects
        self.molecules = []
        self.num_molecules = 0

    def process_from_file(self, filename: str):

        # read dataset into a list where each element has:
        # id, oxyz coordinates, dummy property values
        with open(filename, "r") as f:
            dataset = f.read().strip().split("\n\n")

        # for every molecule chunk
        for data in dataset:
            data = data.strip().split("\n")

            # unpack chunk into molecular id and atom xyz coords
            molecule_id = data[0]
            atom_xyzs = data[1:]

            # initialize buffers, counters
            atom_list, atomic_number_list, atomic_coord_list = [], [], []
            atomic_orbital_list, orbital_coord_list, quantum_number_list = [], [], []
            num_electrons = 0

            # load the 3-dimensional structure data
            for row in atom_xyzs:
                # unpack the row structure
                atom, x, y, z = row.split()

                atomic_number = ATOMICNUMBER_DICT[atom]
                num_electrons += atomic_number

                # append the data to the buffers
                atom_list.append(atom)
                atomic_number_list.append([atomic_number])
                xyz = [float(coord) for coord in [x, y, z]]
                atomic_coord_list.append(xyz)

                # atomic orbitals (basis functions) and principal quantum numbers (q=1,2,3)
                if atomic_number <= 2:
                    aqs = [
                        (atom + "1s" + str(i), 1) for i in range(self.basis_set_outer)
                    ]
                elif atomic_number >= 3 and atomic_number <= 10:
                    aqs = (
                        [(atom + "1s" + str(i), 1) for i in range(self.basis_set_inner)]
                        + [
                            (atom + "2s" + str(i), 2)
                            for i in range(self.basis_set_outer)
                        ]
                        + [
                            (atom + "2p" + str(i), 2)
                            for i in range(self.basis_set_outer)
                        ]
                    )

                elif atomic_number >= 11 and atomic_number <= 12:
                    aqs = (
                        [(atom + "1s" + str(i), 1) for i in range(self.basis_set_inner)]
                        + [
                            (atom + "2s" + str(i), 2)
                            for i in range(self.basis_set_inner)
                        ]
                        + [
                            (atom + "2p" + str(i), 2)
                            for i in range(self.basis_set_inner)
                        ]
                        + [
                            (atom + "3s" + str(i), 3)
                            for i in range(self.basis_set_outer)
                        ]
                    )

                else:  # elif atomic_number >= 13 and atomic_number <= 18:
                    aqs = (
                        [(atom + "1s" + str(i), 1) for i in range(self.basis_set_inner)]
                        + [
                            (atom + "2s" + str(i), 2)
                            for i in range(self.basis_set_inner)
                        ]
                        + [
                            (atom + "2p" + str(i), 2)
                            for i in range(self.basis_set_inner)
                        ]
                        + [
                            (atom + "3s" + str(i), 3)
                            for i in range(self.basis_set_outer)
                        ]
                        + [
                            (atom + "3p" + str(i), 3)
                            for i in range(self.basis_set_outer)
                        ]
                    )

                for a, q in aqs:
                    atomic_orbital_list.append(a)
                    orbital_coord_list.append(xyz)
                    quantum_number_list.append(q)

            # cast into arrays, use auxiliary functions to convert
            atomic_coord_list = np.array(atomic_coord_list)
            atomic_orbital_list = create_orbitals(
                orbitals=atomic_orbital_list, orbital_dict=self.orbital_dict
            )
            field_coordinates = create_field(
                sphere=self.sphere, coords=atomic_coord_list
            )
            distance_matrix = create_distancematrix(
                coords1=field_coordinates, coords2=atomic_coord_list
            )
            atomic_number_list = np.array(atomic_number_list)
            potential = create_potential(
                distance_matrix=distance_matrix, atomic_numbers=atomic_number_list
            )

            distance_matrix = create_distancematrix(
                coords1=field_coordinates, coords2=orbital_coord_list
            )
            quantum_number_list = np.array([quantum_number_list])
            num_electrons = np.array([[num_electrons]])
            num_fields = len(field_coordinates)  # number of points in the grid field

            # wrap all the array together into single list
            molecule = [
                molecule_id,
                atomic_orbital_list.astype(np.int64),
                distance_matrix.astype(np.float32),
                quantum_number_list.astype(np.float32),
                num_electrons.astype(np.float32),
                num_fields,
            ]
            molecule = np.array(molecule, dtype=object)
            self.molecules.append(molecule)
            self.num_molecules += 1

        self.empty = False

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        return self.molecules[idx]
