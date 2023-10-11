import sys

sys.path.append("..")

import gc

import torch

from funnel.generator.vocab import common_atom_vocab
from funnel.generator.wrappers import GeneratorModel
from funnel.surrogate.processor import QDFprocessor, smiles_to_xyz
from funnel.surrogate.wrappers import SurrogateModel


def funnel_half(
    pretrained_generator: str,
    pretrained_surrogate: str,
    motif_vocab: str,
    orbital_file: str,
    device_gen: str,
    device_surr: str,
    cond,
):

    # load the wrappers of the generator and surrogate model
    device = torch.device(device_gen)
    generator = GeneratorModel(
        pretrained=pretrained_generator,
        motif_vocab=motif_vocab,
        atom_vocab=common_atom_vocab,
        device=device,
    )

    # generate the initial set
    generator.generate(num_samples=100, condition=cond, unique=True)
    generated = generator.get_results()

    del generator
    collected = gc.collect()

    # dump the generated smiles into a file
    with open("gen_out.smi", "w") as f:
        [print(smile, file=f) for smile in generated]

    # transform the smiles file to xyz
    smiles_to_xyz(in_filename="gen_out.smi", out_filename="surr_in.xyz")

    # load the dataset with the xyz smiles
    dataset = QDFprocessor(orbital_file=orbital_file)
    dataset.process_from_file(filename="surr_in.xyz")

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        32,
        shuffle=False,
        num_workers=1,
        collate_fn=lambda xs: list(zip(*xs)),
        pin_memory=True,
    )

    del dataset
    collected = gc.collect()

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

    with open("calibration_script_output.txt", "a+") as f:
        for mol, pred in zip(molecule_ids, predictions):
            print(mol, cond, pred, file=f)


if __name__ == "__main__":

    import sys

    pretrained_generator = "../pretrained/pretrained_generator"
    pretrained_surrogate = "../pretrained/pretrained_surrogate"
    motif_vocab = "../configuration/vocab.txt"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"

    # runs the half pipeline (Generator + Surrogate)
    funnel_half(
        pretrained_generator=pretrained_generator,
        pretrained_surrogate=pretrained_surrogate,
        motif_vocab=motif_vocab,
        orbital_file=orbital_file,
        device_gen="cuda" if torch.cuda.is_available() else "cpu",
        device_surr="cpu",
        cond=float(sys.argv[1]),
    )
