import sys

sys.path.append("..")

from funnel.surrogate.processor import QDFprocessor


def test_fileprocessing_QDFprocessor():
    """
    Test the QDFprocessor object can run the process_file method without errors/exceptions
    """
    processor = QDFprocessor(orbital_file="../configuration/orbitaldict_6-31G.pickle")
    processor.process_from_file(filename="./data/samples_smiles2xyz.xyz")


def test_numouts_fileprocessing_QDFprocessor():
    """
    Test the QDFprocessor object can run the process_file method and this results in as many processed
    outputs as raw inputs (we didnt miss any molecule given)
    """
    processor = QDFprocessor(orbital_file="../configuration/orbitaldict_6-31G.pickle")
    processor.process_from_file(filename="./data/samples_smiles2xyz.xyz")
    assert len(processor) == 19
