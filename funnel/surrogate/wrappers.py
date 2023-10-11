import pickle

import torch

from funnel.surrogate.hyperparameters import (
    DIM,
    HIDDEN_HK,
    LAYER_FUNCTIONAL,
    LAYER_HK,
    OPERATION,
)
from funnel.surrogate.models import QuantumDeepField
from funnel.surrogate.processor import QDFprocessor, smiles_to_xyz


class SurrogateModel(object):
    """
    Wrapper object for performing the inference on the Surrogate
    QuantumDeepField model

    :param str pretrained: String path to the filename with the pre-trained surrogate model (state dictionary).
    :param str orbital_file: String path to the .pickle file containing the orbital dictionary, as produces during the data processing.
    :param torch.device device: Torch.device ('cuda' or 'cpu') to be used for inference.
    """

    def __init__(self, pretrained: str, orbital_file: str, device: torch.device):

        # Load orbital_dict generated in preprocessing.
        with open(orbital_file, "rb") as f:
            orbital_dict = pickle.load(f)
        N_orbitals = len(orbital_dict)

        # create and load the pretrained generator
        self.device = device
        self.model = QuantumDeepField(
            device=device,
            N_orbitals=N_orbitals,
            dim=DIM,
            layer_functional=LAYER_FUNCTIONAL,
            operation=OPERATION,
            N_output=2,
            hidden_HK=HIDDEN_HK,
            layer_HK=LAYER_HK,
        ).to(device)
        model_state = torch.load(pretrained, map_location=device)
        self.model.load_state_dict(model_state)

        # set to evaluation mode
        self.model.eval()

        # initialize the buffers
        self.molecule_id_buffer, self.prediction_buffer = [], []

    def forward(self, batch):
        """
        Wrapper for calling the forward method on the surrogate model and computing
        the homo-lumo gap from the outputted predictions. It returns the list of molecular_ids (molecule SMILES) and
        list of gap predicitons (in eV). During inference all the predicitons get stored in an internal buffer that can then be prompted
        using the get_results() method.

        :param batch: Batch of samples to make inference on.
        :returns: tuple of 2 lists, list containing the molecular ids (SMILES strings) to be used in identifying the predicitons, and list of homo-lumo gap predictions.
        :rtype: tuple
        """
        # predict=True untracks the gradients already
        molecules_ids, prediction = self.model.forward(batch, predict=True)

        # compute the homo-lumo gap
        prediction = torch.sub(prediction[:, 1], prediction[:, 0]).cpu().tolist()

        # Note this '+' operation on lists denotes concatenation
        self.molecule_id_buffer += molecules_ids
        self.prediction_buffer += prediction
        return molecules_ids, prediction

    def get_results(self):
        """
        Getter method to return the list of molecular ids and predicitons generated during the whole inference process.

        :returns: tuple of 2 lists, list containing the molecular ids (SMILES strings) to be used in identifying the predicitons, and list of homo-lumo gap predictions.
        :rtype: tuple
        """
        return self.molecule_id_buffer, self.prediction_buffer
