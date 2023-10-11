import sys

sys.path.append("..")

from funnel.surrogate.processor import QDFprocessor, smiles_to_xyz


def test_integration_surrogate_feeder():
    """
    Test the integration of smiles_to_xyz + QDF_processor.
    Verify the number of inputs and outputs are the same.
    Verify the same molecules appear in the inputs and outputs
    """

    generator_out = "./data/samples_smiles.smi"
    surrogate_in = "./data/samples_smiles2xyz.xyz"

    smiles_to_xyz(in_filename=generator_out, out_filename=surrogate_in)

    with open(generator_out, "r") as f:
        smiles = f.read().strip().split("\n")

    processor = QDFprocessor(orbital_file="../configuration/orbitaldict_6-31G.pickle")
    processor.process_from_file(filename=surrogate_in)
    assert len(processor) == len(smiles)

    for smile, molecule in zip(smiles, processor.molecules):
        assert smile == molecule[0]
