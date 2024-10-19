import pandas as pd
import shutil
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import click
from functools import partial

import json
import warnings

import numpy as np
from pymatgen.core.structure import Structure

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(
            -((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2
        )


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def process(cifid: str, root_dir: Path):
    cif_dir = root_dir / 'cif'
    tgt_dir = root_dir / 'cgcnn'

    cif_path = cif_dir / f"{cifid}.cif"
    max_num_nbr = 12
    radius = 8
    atom_init_file = "./data/sample-regression/atom_init.json"
    ari = AtomCustomJSONInitializer(atom_init_file)
    dmin = 0
    step = 0.2
    gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
    # TODO: cifid, target = id_prop[idx]
    crystal = Structure.from_file(cif_path.absolute())
    atom_fea = np.vstack(
        [ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
    )
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn(
                "{} not find enough neighbors to build graph. "
                "If it happens frequently, consider increase "
                "radius.".format(cifid)
            )
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                list(map(lambda x: x[1], nbr))
                + [radius + 1.0] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)

    np_data = {"atom_fea": atom_fea, "nbr_fea": nbr_fea, "nbr_fea_idx": nbr_fea_idx}
    np.save((tgt_dir / f"{cifid}.npy").absolute(), np_data)



# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", type=str)
# parser.add_argument("--task", type=str)
# parser.add_argument("--type", type=str, default="cif")
# args = parser.parse_args()


@click.command()
@click.option('--root-dir', '-R', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--n-process', '-N', type=int, default=8)
@click.option('--task', '-T', type=str)
def main(root_dir, n_process, task):
    splits = ['train', 'val', 'test']
    idprop_paths = {
        split: root_dir / f"benchmark.{split}.csv"
        for split in splits
    }
    cif_dir = root_dir / 'cif'
    tgt_dir = root_dir / "cgcnn"
    cifids = [f.stem for f in cif_dir.iterdir() if f.suffix == '.cif']


    if not tgt_dir.exists():
        tgt_dir.mkdir(parents=True)


    print('START TO BUILD MODAL DATA')
    with Pool(n_process) as pool:
        result = list(tqdm(pool.imap_unordered(partial(process, root_dir=root_dir), cifids), total=len(cifids)))

    print("START TO PROCESS ID_PROP FILE")
    for split in splits:
        idprop_df = pd.read_csv(idprop_paths[split].absolute(), dtype={
            "cifid": str
        })
        idprop_df = idprop_df.dropna(subset=[task])
        items = []
        for item in tqdm(idprop_df.iloc, total=len(idprop_df)):
            items.append([item["cifid"], item[task]])
        new_df = pd.DataFrame(items)
        new_df.to_csv(tgt_dir / f"id_prop.{split}.csv", index=False, header=False)

if __name__ == '__main__':
    main()
