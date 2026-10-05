import numpy as np
from rdkit import Chem

allowed_bonds = {
    'H': {1}, 'C': {4}, 'N': {3}, 'O': {2}, 'F': {1},
    'B': {3}, 'Al': {3}, 'Si': {4}, 'P': {5, 3},
    'S': {6, 4, 2}, 'Cl': {1},
    'As': {3}, 'Br': {1}, 'I': {1},
    'Hg': {2}, 'Bi': {3}
}

bond_dict = {'H': {1: 0.74},
             'C': {1: 1.54, 2: 1.34, 3: 1.20},
             'N': {1: 1.47, 2: 1.25, 3: 1.10},
             'O': {1: 1.48, 2: 1.21},
             'F': {1: 1.42}}

def get_bond_order(atom1, atom2, distance, tolerance=0.45):
    bond_order = 0
    if atom1 not in bond_dict or atom2 not in bond_dict:
        return bond_order

    dist_dict = bond_dict[atom1]
    for order, dist in dist_dict.items():
        if abs(distance - dist) < tolerance:
            bond_order = order
            break

    dist_dict = bond_dict[atom2]
    for order, dist in dist_dict.items():
        if abs(distance - dist) < tolerance:
            bond_order = max(bond_order, order)
            break
    return bond_order

ATOMIC_NUMBER_TO_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 13: 'Al', 14: 'Si', 
    15: 'P', 16: 'S', 17: 'Cl', 33: 'As', 35: 'Br', 53: 'I', 80: 'Hg', 83: 'Bi' 
}

def check_stability(positions, atom_type_integers, dataset_info):
    positions = positions.cpu().numpy()
    atom_type_integers = atom_type_integers.cpu().numpy()
    
    num_atoms = len(atom_type_integers)
    if num_atoms == 0:
        return {'is_stable': False, 'stability_score': 0.0, 'invalid_atoms_report': ['Molecule has no atoms.']}

    nr_bonds = np.zeros(num_atoms, dtype=int)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            atom1_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atom_type_integers[i])
            atom2_symbol = ATOMIC_NUMBER_TO_SYMBOL.get(atom_type_integers[j])
            if atom1_symbol is None or atom2_symbol is None: continue
            order = get_bond_order(atom1_symbol, atom2_symbol, dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    invalid_atoms_report = []
    for i in range(num_atoms):
        atom_num = atom_type_integers[i]
        atom_name = ATOMIC_NUMBER_TO_SYMBOL.get(atom_num)
        if atom_name is None: continue

        possible_bonds = allowed_bonds.get(atom_name)
        if possible_bonds is None: continue

        is_atom_stable = nr_bonds[i] in possible_bonds
        if is_atom_stable:
            nr_stable_bonds += 1
        else:
            report = f"Atom #{i} ({atom_name}): Expects valence in {possible_bonds}, but got {nr_bonds[i]}."
            invalid_atoms_report.append(report)
    
    stability_score = nr_stable_bonds / num_atoms
    is_molecule_stable = (nr_stable_bonds == num_atoms)
    
    return {
        'is_stable': is_molecule_stable,
        'stability_score': stability_score,
        'invalid_atoms_report': invalid_atoms_report
    }