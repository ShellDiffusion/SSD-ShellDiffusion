import os
import sys
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict
import numpy as np
from collections import Counter
from torch_geometric.data import Data, Batch



from rdkit import Chem
from qm9.bond_analyze import get_bond_order
from configs.datasets_config import get_dataset_info

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from models.epsnet.dualenc import center_pos, get_distance
from models.geometry import eq_transform

ATOMIC_NUMBER_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
def build_molecule(positions, atom_types):
    mol = Chem.RWMol()
    for atomic_num in atom_types:
        mol.AddAtom(Chem.Atom(int(atomic_num.item())))
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(len(positions)):
        for j in range(i):
            atom_symbol_i = ATOMIC_NUMBER_TO_SYMBOL.get(atom_types[i].item())
            atom_symbol_j = ATOMIC_NUMBER_TO_SYMBOL.get(atom_types[j].item())
            if atom_symbol_i and atom_symbol_j:
                order = get_bond_order(atom_symbol_i, atom_symbol_j, dists[i, j])
                if order > 0: mol.AddBond(i, j, bond_dict[order])
    return mol

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except: return None

def is_chemically_reasonable(mol):
 
    if mol is None:
        return False, "Molecule object is None"
    
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        
        if '.' in smiles:
            return False, "Molecule consists of disconnected fragments"
            
        return True, "OK"
    except (ValueError, RuntimeError) as e:
        return False, f"Sanitization error: {e}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--tag', type=str, default='unconditional_filtered')
    args = parser.parse_args()


    ckpt = torch.load(args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f: config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)
    logger.info('Loading model...')
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    test_set = ConformationDataset(config.dataset.test)
    size_distribution = [data.num_nodes for data in test_set]
    all_atom_types = torch.cat([data.atom_type for data in test_set]).tolist()
    atom_type_counts = Counter(all_atom_types)
    atom_types, atom_counts = zip(*atom_type_counts.items())
    atom_probabilities = np.array(atom_counts) / sum(atom_counts)
    dataset_info = get_dataset_info('qm9', remove_h=False)

    results = []
    pbar = tqdm(total=args.num_samples, desc='Generating Molecules')
    attempts = 0 
    
    while len(results) < args.num_samples:
        num_candidates = args.batch_size * 10 
        
        num_nodes_list = np.random.choice(size_distribution, size=num_candidates)
        
        valid_atom_type_list = []
        valid_num_nodes_list = []
        for num_nodes in num_nodes_list:
            attempts += 1
            sampled_types = np.random.choice(atom_types, size=num_nodes, p=atom_probabilities)
            
            mol_from_composition = Chem.RWMol()
            for atomic_num in sampled_types:
                mol_from_composition.AddAtom(Chem.Atom(int(atomic_num)))
            
            try:
                Chem.SanitizeMol(mol_from_composition)
                valid_atom_type_list.append(torch.from_numpy(sampled_types).long())
                valid_num_nodes_list.append(num_nodes)
            except (ValueError, RuntimeError):
                continue

        if not valid_atom_type_list:
            continue

        current_batch_size = len(valid_atom_type_list)
        atom_type_sampled = torch.cat(valid_atom_type_list, dim=0).to(args.device)
        batch_indices = []
        for i, num_nodes in enumerate(valid_num_nodes_list):
            batch_indices.append(torch.full((num_nodes,), fill_value=i, dtype=torch.long))
        batch = torch.cat(batch_indices, dim=0).to(args.device)
        pos_init = torch.randn(atom_type_sampled.size(0), 3).to(args.device)

        try:
            with torch.no_grad():
                pos_gen, _ = model.langevin_dynamics_sample_diffusion(
                    atom_type=atom_type_sampled,
                    pos_init=pos_init,
                    bond_index=torch.empty(2, 0, dtype=torch.long, device=args.device),
                    bond_type=torch.empty(0, dtype=torch.long, device=args.device),
                    batch=batch,
                    num_graphs=current_batch_size,
                    extend_order=False,
                    extend_radius=True,
                    n_steps=args.n_steps
                )

            pos_list = [pos_gen[batch == i].cpu() for i in range(current_batch_size)]
            type_list = [atom_type_sampled[batch == i].cpu() for i in range(current_batch_size)]

            for pos, types in zip(pos_list, type_list):
                mol = build_molecule(pos, types)
                is_valid, reason = is_chemically_reasonable(mol)
                
                if is_valid:
                    smiles = Chem.MolToSmiles(mol)
                    data = Data(pos_gen=pos, atom_type=types, smiles=smiles, rdmol=mol)
                    results.append(data)
                    pbar.update(1)

                    current_sample_index = len(results) - 1
                    save_path_intermediate = os.path.join(output_dir, f'samples_{current_sample_index}.pkl')
                    logger.info(f'Success! Saving intermediate results for molecule #{current_sample_index} to: {save_path_intermediate}')
                    with open(save_path_intermediate, 'wb') as f:
                        pickle.dump(results, f)

                    if len(results) >= args.num_samples:
                        break 
            
            if len(results) >= args.num_samples:
                break 
        
        except FloatingPointError:
            logger.warning('NaN detected, skipping batch.')
        except Exception as e:
            logger.error(f"Error Msg: {e}")

    pbar.close()

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info(f'Saving {len(results)} generated molecules to: {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
