import sys

sys.path.append("..")

from funnel.surrogate.processor import smiles_to_xyz


def test_pass_obabel_smiles_to_xyz():
    """
    Test the function runs without errors/exceptions.
    """
    smiles_to_xyz(
        in_filename="./data/samples_smiles.smi",
        out_filename="./data/samples_smiles2xyz.xyz",
    )


def test_outs_obabel_smiles_to_xyz():
    """
    Test that there is as many molecules in the outputs as in the inputs, then test if they individually
    match the input molecules.
    """
    smiles_to_xyz(
        in_filename="./data/samples_smiles.smi",
        out_filename="./data/samples_smiles2xyz.xyz",
    )

    with open("./data/samples_smiles2xyz.xyz", "r") as f:
        chunks = f.read().strip().split("\n\n")

    with open("./data/samples_smiles.smi", "r") as f:
        smiles = f.read().strip().split("\n")

    assert len(smiles) == len(chunks)

    for smile, chunk in zip(smiles, chunks):
        assert smile == chunk.split("\n")[0]
