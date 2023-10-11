import sys

sys.path.append("..")

import torch

from funnel.surrogate.processor import QDFprocessor
from funnel.surrogate.wrappers import SurrogateModel


def test_intitialization_surrogate():
    """
    Tests if we can load the pretrained surrogate model into
    the wrapping class.
    """
    pretrained = "../pretrained/pretrained_surrogate"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurrogateModel(
        pretrained=pretrained, orbital_file=orbital_file, device=device
    )


def test_integration_surrogate():
    """
    Tests the integration of the surrogate processor and the surrogate.
    Reading some xyz molecules, the processor shapes them into the format readable by the surrogate,
    then the surrogate makes the predictions. Then we check at the end that the molecules at the end are the same
    that there were at the beggining by checking the SMILES strings.
    """
    pretrained = "../pretrained/pretrained_surrogate"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurrogateModel(
        pretrained=pretrained, orbital_file=orbital_file, device=device
    )

    dataset = QDFprocessor(orbital_file=orbital_file)
    dataset.process_from_file("./data/samples_smiles2xyz.xyz")
    datalaoder = torch.utils.data.DataLoader(
        dataset,
        3,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda xs: list(zip(*xs)),
        pin_memory=True,
    )

    for data in datalaoder:
        molecules_ids, prediction = model.forward(batch=data)
        break

    with open("./data/samples_smiles.smi", "r") as f:
        smiles = f.read().strip().split("\n")

    for molecule_id, smile in zip(molecules_ids, smiles):
        assert molecule_id == smile


def test_integration_surrogate_getter_method():
    """
    Tests the integration of the surrogate processor and the surrogate.
    Reading some xyz molecules, the processor shapes them into the format readable by the surrogate,
    then the surrogate makes the predictions. This time we check the getter method,
    that returns all the predictions done by the surrogate.
    Then we check at the end that the molecules at the end are the same
    that there were at the beggining by checking the SMILES strings.
    """
    pretrained = "../pretrained/pretrained_surrogate"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SurrogateModel(
        pretrained=pretrained, orbital_file=orbital_file, device=device
    )

    dataset = QDFprocessor(orbital_file=orbital_file)
    dataset.process_from_file("./data/samples_smiles2xyz.xyz")
    datalaoder = torch.utils.data.DataLoader(
        dataset,
        3,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda xs: list(zip(*xs)),
        pin_memory=True,
    )

    for data in datalaoder:
        _, _ = model.forward(batch=data)

    with open("./data/samples_smiles.smi", "r") as f:
        smiles = f.read().strip().split("\n")

    molecule_ids, predictions = model.get_results()

    for molecule_id, smile in zip(molecule_ids, smiles):
        assert molecule_id == smile
