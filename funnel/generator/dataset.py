import gc
import os
import pickle
import random

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from funnel.generator.chemutils import get_leaves
from funnel.generator.mol_graph import MolGraph


class MoleculeDataset(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        safe_data = []
        for mol_s in data:
            hmol = MolGraph(mol_s)
            ok = True
            for node, attr in hmol.mol_tree.nodes(data=True):
                smiles = attr["smiles"]
                ok &= attr["label"] in vocab.vmap
                for i, s in attr["inter_label"]:
                    ok &= (smiles, s) in vocab.vmap
            if ok:
                safe_data.append(mol_s)

        print(f"After pruning {len(data)} -> {len(safe_data)}")
        self.batches = [
            safe_data[i : i + batch_size] for i in range(0, len(safe_data), batch_size)
        ]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


class MolEnumRootDataset(Dataset):
    def __init__(self, data, vocab, avocab):
        self.batches = data
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = set(
            [
                Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False)
                for i in leaves
            ]
        )
        smiles_list = sorted(list(smiles_list))  # To ensure reproducibility

        safe_list = []
        for s in smiles_list:
            hmol = MolGraph(s)
            ok = True
            for node, attr in hmol.mol_tree.nodes(data=True):
                if attr["label"] not in self.vocab.vmap:
                    ok = False
            if ok:
                safe_list.append(s)

        if len(safe_list) > 0:
            return MolGraph.tensorize(safe_list, self.vocab, self.avocab)
        else:
            return None


class DataFolder(object):
    """
    Class for loading batches of data from the .pkl files produced in the preprocessing
    given the path to the folder where the .pkl files are stored.
    """

    def __init__(self, data_folder: str, batch_size: int, shuffle: bool = True):
        """
        :param data_folder: str path to folder where the .pkl files containing the prepreocess data are stored
        :param batch_size: int number of samples to load per batch NOTE: its useless since the batches depend on how the preprocessing was done
        nothing changes if this nuber is changed
        :param shuffle: boolean flag to shuffle the batyches once loaded. Default True.
        """
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # compute the length of the dataset
        n_batches = 0
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                batches = pickle.load(f)
                n_batches += len(batches)
                del batches
                gc.collect()
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                batches = pickle.load(f)

            if self.shuffle:
                random.shuffle(batches)  # shuffle data before batch

            for batch in batches:
                yield batch

            del batches
            gc.collect()
