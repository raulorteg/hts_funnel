import sys

sys.path.append("..")

import gc

import torch
from funnel.configuration import THRESHOLD_SURROGATE
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
    cond: float,
    num_samples: int,
    output_filename: str,
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
    generator.generate(num_samples=num_samples, condition=cond, unique=True)
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

    ctr_initial, ctr_final = 0, 0
    with open(output_filename, "a+") as f:
        for mol, pred in zip(molecule_ids, predictions):
            ctr_initial += 1
            if abs(pred - args.cond) < THRESHOLD_SURROGATE:
                ctr_final += 1
                print(mol, cond, pred, file=f)

    # message for user summarizing results
    print("---------------------------------- ")
    print(f"------ Summary of results ------ ")
    print("---------------------------------- ")
    print(f"\t - Input condition: {args.cond} eV HOMO-LUMO energy gap.")
    print(f"\t - Generator created: {ctr_initial} candidates.")
    print(f"\t - Surrogate curated the candidates into: {ctr_final} candidates.")
    print(
        f"\t - The final candidates are saved in SMILES format in {args.fout}, together with their predicted HOMO-LUMO gaps (eV)."
    )
    print("---------------------------------- ")


if __name__ == "__main__":

    import argparse
    import sys

    pretrained_generator = "../pretrained/pretrained_generator"
    pretrained_surrogate = "../pretrained/pretrained_surrogate"
    motif_vocab = "../configuration/vocab.txt"
    orbital_file = "../configuration/orbitaldict_6-31G.pickle"

    # parse user input (Number of inital samples to create, condition for the generation process, file to dump the results at.)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=100,
        help="Number of initial Generated samples to create. Defaults to 100.",
    )
    parser.add_argument(
        "--cond",
        type=float,
        required=True,
        help="HOMO-LUMO gap (eV) condition for the generation process (Target).",
    )
    parser.add_argument(
        "--fout",
        type=str,
        required=False,
        default="../data/results_funnel.txt",
        help="Filename to save the outputs of the funnel process. Defatuls to <results_funnel.txt>.",
    )
    args = parser.parse_args()

    # runs the half pipeline (Generator + Surrogate)
    funnel_half(
        pretrained_generator=pretrained_generator,
        pretrained_surrogate=pretrained_surrogate,
        motif_vocab=motif_vocab,
        orbital_file=orbital_file,
        device_gen="cuda" if torch.cuda.is_available() else "cpu",
        device_surr="cpu",
        cond=args.cond,
        num_samples=args.num_samples,
        output_filename=args.fout,
    )
