import gc
import sys

sys.path.append("..")

import torch

from funnel.generator.vocab import common_atom_vocab
from funnel.generator.wrappers import GeneratorModel
from funnel.surrogate.processor import QDFprocessor, smiles_to_xyz
from funnel.surrogate.wrappers import SurrogateModel


def test_half_pipeline():
    """
    Tests the integration of the half pipeline:
        Generator + obabel + Surrogate processor + Surrogate
    by generating 10 initial molecules, then predicting its HOMO-LUMO gaps
    using the surrogate model.
    """

    # inputs for the half-pipeline to run
    pretrained_generator = "../pretrained/pretrained_generator"
    pretrained_surrogate = "../pretrained/pretrained_surrogate"
    motif_vocab = "../configuration/vocab.txt"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"
    device_gen = "cuda" if torch.cuda.is_available() else "cpu"
    device_surr = "cpu"
    cond = 2.0

    # load the wrapper for the generator
    device = torch.device(device_gen)
    generator = GeneratorModel(
        pretrained=pretrained_generator,
        motif_vocab=motif_vocab,
        atom_vocab=common_atom_vocab,
        device=device,
    )

    # generate the initial set (dont check for uniqueness since takes time)
    generator.generate(num_samples=10, condition=cond, unique=False)
    generated = generator.get_results()

    # cleanup
    del generator
    collected = gc.collect()

    print("Garbage collector: collected", "%d objects." % collected)

    # dump the generated smiles into a file
    with open("./data/gen_out.smi", "w") as f:
        [print(smile, file=f) for smile in generated]

    # transform the smiles file to xyz
    smiles_to_xyz(in_filename="./data/gen_out.smi", out_filename="./data/surr_in.xyz")

    # load the dataset with the xyz smiles
    dataset = QDFprocessor(orbital_file=orbital_file)
    dataset.process_from_file(filename="./data/surr_in.xyz")

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        32,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda xs: list(zip(*xs)),
        pin_memory=True,
    )

    # cleanup
    del dataset
    collected = gc.collect()

    # load the wrapper for the surrogate model
    device = torch.device(device_surr)
    surrogate = SurrogateModel(
        pretrained=pretrained_surrogate,
        orbital_file=orbital_file,
        device=device,
    )

    # feed the data to the surrogate model
    for data in dataloader:
        _, _ = surrogate.forward(batch=data)

    # get the results from the internal buffers
    molecule_ids, predictions = surrogate.get_results()
