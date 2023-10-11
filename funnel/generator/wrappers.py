import rdkit
import torch

from funnel.generator.hgnn import CondHierVAE
from funnel.generator.vocab import PairVocab, common_atom_vocab


class GeneratorModel(object):
    """
    Wrapper object for the generator. It wraps up the generation to simplify its use.

    :param str pretrained: String path to the filename with the pre-trained generator model (state dictionary).
    :param str motif_vocab: String path to the vocabulary motif file.
    :param atom_vocab: Atom vocabulary dictionary
    :param torch.device device: Torch.device ('cuda' or 'cpu') to be used for inference.
    """

    def __init__(
        self, pretrained: str, motif_vocab: str, atom_vocab, device: torch.device
    ):

        # load the motif vocabulary from file
        motif_vocab = [x.strip("\r\n ").split() for x in open(motif_vocab)]
        motif_vocab = PairVocab(motif_vocab, cuda=bool(device == torch.device("cuda")))

        # create the empty model with the motif and atom vocabularies and load the state dictionary
        self.model = CondHierVAE(
            device=device, vocab=motif_vocab, atom_vocab=atom_vocab
        ).to(device)
        model_state, _, _, _, _ = torch.load(pretrained, map_location=device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        self.device = device

        # initialize buffers and counters for generation
        self.generated = []
        self.num_generated = 0

    def __len__(self):
        return self.num_generated

    def generate(self, num_samples: int, condition: float, unique: bool = True):
        """
        Main method of the GeneratorModel object. Wraps the generation of the molecules.
        Given a number of samples, a condition and a requirement for uniqueness the method will generate
        the molecules according to the condition and will check for uniqueness on them.

        :param int num_samples: Number of samples to generate.
        :param float condition: Condition for the molecules generated (homo-lumo desired gap value, in eV)
        :param bool unique: Bolean flag to ensure that the molecules returned are not repeated. (It slows down generation, since it checks for repetitions)
        :returns: list of SMILES string with the molecules generated. Note if the flag unqieu is activated these will be Canonical SMILES.
        :rtype: list
        """
        with torch.no_grad():
            while self.num_generated < num_samples:

                # generate a number <batch_size> of molecules with the generator
                molecules = self.model.sample_cond(
                    batch_size=32, greedy=False, cond=condition
                )

                for molecule in molecules:

                    # if unique flag is forced, then we only add them if
                    # they had not been generated before
                    if unique:

                        # compute its canonical SMILES
                        can_molecule = rdkit.Chem.CanonSmiles(molecule)

                        # check if in buffer
                        if can_molecule not in self.generated:
                            self.generated.append(can_molecule)
                            self.num_generated += 1

                    # else we will append them to the list of generated,
                    # even if they were already there
                    else:
                        self.generated.append(molecule)
                        self.num_generated += 1

            # if we generate more than asked then remove the last few
            if self.num_generated > num_samples:
                self.generated = self.generated[: -(self.num_generated - num_samples)]
                self.num_generated = num_samples

        return self.generated

    def get_results(self):
        """
        Getter method to return the list of molecular SMILES (canonical) generated.

        :returns: list containing the molecular (SMILES strings) of the molecules generated.
        :rtype: list
        """
        return self.generated

    def reset(self):
        """
        Method to restart the buffer and generation counter.
        """
        self.generated = []
        self.num_generated = 0
