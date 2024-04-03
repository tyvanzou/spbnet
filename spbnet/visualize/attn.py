import argparse
from pathlib import Path
from spbnet.utils.echo import title, start, end
from .buildModalData import buildModalData
from spbnet.modules.module import CrossFormer
from .utils import get_grid_data, get_graph, collate, get_atoms
import numpy as np
from pathlib import Path
from subprocess import Popen
# from rdkit import Chem
import json
from ase.io import read, write

cur_dir = Path(__file__).parent


def attn(cif_path: Path, modal_dir: Path, ckpt: Path, out_dir: Path):
    title("START TO GET ATTENTION WEIGHT")
    cifid = cif_path.stem

    percentile = 50
    # ratio = 3

    start("Start to prepare data")

    buildModalData(cif_path, modal_dir)

    end("Prepare modal data end")

    import torch
    import yaml

    start(f"Start to load weight from {ckpt}")

    ckpt = torch.load(
        ckpt,
        # f"./ckpt/p10w/{probe}.ckpt",
        map_location="cpu",
    )
    state_dict = ckpt["state_dict"]
    for key in list(state_dict.keys()):
        if not key.startswith("model."):
            del state_dict[key]
    state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}
    with open((cur_dir / "config.yaml").absolute(), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["visualize"] = True
    model = CrossFormer(config=config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    end("Weight load success")

    print("Start to get atttention weight")

    data_dir = modal_dir

    batch = []
    item = dict()
    item.update({"cifid": cifid, "target": 0})
    item.update(get_grid_data(data_dir, cifid))
    graph_data = get_graph(data_dir, cifid)
    item.update(graph_data)
    atom_num = min(graph_data["atom_num"].shape[0], 512)
    atom_idx = graph_data["atom_num"]
    atom_idx = atom_idx.detach().numpy()
    batch.append(item)
    batch = collate(batch)
    out = model(batch)
    sa_attn: torch.Tensor = out["sa_attn"]  # [B, lj_len, lj_len]
    ma_attn: torch.Tensor = out["ca_attn"]  # [B, lj_len, graph_len]

    potential_attn = sa_attn[0, 0].detach().numpy()
    # potential_attn = potential_attn[2:-2].detach().numpy()
    atom_attn = ma_attn[0, 0].detach().numpy()  # [graph_len]
    top_potential_indices = np.array(np.argsort(potential_attn)[-1:])
    # max_potential_attn_idx = np.argmax(potential_attn)
    # potential_atom_attn = ma_attn[0, int(max_potential_attn_idx)].detach().numpy()
    potential_atom_attn = ma_attn[0, top_potential_indices].detach().numpy()
    # potential_atom_attn = np.concatenate(
    #     [potential_atom_attn, atom_attn.reshape(1, -1)], axis=0
    # )
    potential_atom_attn = np.max(potential_atom_attn, axis=0, keepdims=False)
    # potential_atom_attn.shape, potential_atom_attn[:10], atom_attn.shape, atom_attn[
    #     :10
    # ], max_potential_attn_idx
    end(
        f"Get attention weight success, Max: {np.max(atom_attn)}, Min: {np.min(atom_attn)}, Mean: {np.mean(atom_attn)}"
    )

    start("Start to construct mol data")

    # proc = Popen(
    #     [
    #         "obabel",
    #         (modal_dir / f"supercell/{cifid}.cif").absolute(),
    #         "-O",
    #         (modal_dir / f"mol/{cifid}.mol").absolute(),
    #         "--addhydrogens",
    #     ]
    # )
    # proc.wait()
    xyz_atoms = read((modal_dir / f"supercell/{cifid}.cif").absolute())
    write((modal_dir / f"mol/{cifid}.xyz").absolute(), xyz_atoms, format="xyz")

    end("Construct mol data success")

    start("Start to get atoms")
    atoms = get_atoms((modal_dir / f"supercell/{cifid}.cif").absolute())
    atom_positions = atoms.get_positions()
    end(f"Supercell, atom num: {len(atom_positions)}")

    # # 读取MOL文件
    # mol_file_path = (modal_dir / "mol" / f"{cifid}.mol").absolute()
    # mol = Chem.MolFromMolFile(mol_file_path, sanitize=False)

    # # 获取原子坐标
    # conf = mol.GetConformer()
    # mol_positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    # len(mol_positions), mol_positions[:10]
    # end(f"Mol data got, atom num: {len(mol_positions)}")

    # start(f"Use template from ./template/attn.html")
    start("Start to build attention html")

    vis_attn: np.array = atom_attn
    vis_attn = vis_attn[:atom_num]
    atom_idx = atom_idx[:atom_num]
    mean, std = np.mean(vis_attn), np.std(vis_attn)
    vis_attn[np.where(atom_idx == 1)] = mean
    # ratio = 3 / (std / mean)
    # attn_mean = np.mean(vis_attn)
    # vis_attn /= attn_mean
    # vis_attn **= ratio
    # vis_attn[vis_attn > 4] = 4
    jsonData = [float(vis_attn[i]) for i in range(min(512, len(atom_positions)))]
    with open(modal_dir / f"attn/{cifid}.json", "w") as f:
        json.dump(jsonData, f)

    attn_html = None
    with open(cur_dir / "template/attn.html", "r") as f:
        attn_html = f.read()
    xyzData_str = None
    attn_str = None
    with open(modal_dir / f"mol/{cifid}.xyz", "r") as f:
        xyzData_str = f.read()
    with open(modal_dir / f"attn/{cifid}.json", "r") as f:
        attn_str = f.read()
    attn_html = attn_html.replace("__xyzData__", xyzData_str)
    attn_html = attn_html.replace("__attn__", attn_str)

    out_dir.mkdir(exist_ok=True)

    with open((out_dir / f"{cifid}.html").absolute(), "w") as f:
        f.write(attn_html)

    end("Visualization success")

    title("GOT ATTENTION SUCCESS")
