import sys

sys.path.append("..")

import torch

from funnel.generator.vocab import common_atom_vocab
from funnel.generator.wrappers import GeneratorModel


def test_initialize():
    """
    Tests if we can load the pretrained model on the Wrapping class for
    the generator
    """

    pretrained = "../pretrained/pretrained_generator"
    motif_vocab = "../configuration/vocab.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorModel(
        pretrained=pretrained,
        motif_vocab=motif_vocab,
        atom_vocab=common_atom_vocab,
        device=device,
    )


def test_numgenerations():
    """
    Tests if we can generate (non-unique since its faster because we dont check)
    different numbers of molecules with the generator.
    """

    pretrained = "../pretrained/pretrained_generator"
    motif_vocab = "../configuration/vocab.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = GeneratorModel(
        pretrained=pretrained,
        motif_vocab=motif_vocab,
        atom_vocab=common_atom_vocab,
        device=device,
    )
    for num_samples in [1, 10, 33]:
        generator.generate(num_samples=num_samples, condition=2.0, unique=False)
        generated = generator.get_results()

        assert len(generated) == num_samples

        generator.reset()
