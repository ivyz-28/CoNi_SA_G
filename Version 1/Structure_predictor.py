from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import get_slab_regions
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase import Atom

from fairchem.core import pretrained_mlip, FAIRChemCalculator

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os

single_atom = None # Replace with desired single atom symbol

struct = Structure.from_file('f./CoNi.cif')

predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cpu") # Change to cuda if running on gpu
calc = FAIRChemCalculator(predictor, task_name="oc20")

# Miller indices
mi=((1,0,0),(1,1,0),(1,1,1),(2,1,0),(2,1,1),(2,2,1),(3,1,0),(3,1,1),(3,2,0),(3,2,1),(3,2,2),(3,3,1),(3,3,2))

adsorbate = Molecule([single_atom],[[0,0,0]])

os.makedirs(f"{single_atom} Fairchem", exist_ok=True)

# Initialize SlabGenerator
for index in mi:
    slabs_gen = SlabGenerator(struct, miller_index=index, min_slab_size=5, min_vacuum_size=20)
    all_slabs = slabs_gen.get_slabs()

    # Write to file within python folder
    for i, slab in enumerate(all_slabs):

        # Make supercell of structure
        slab.make_supercell([2,2,1])

        # Find adsorbate sites for SAC
        asf=AdsorbateSiteFinder(slab)
        sites=asf.find_adsorption_sites()
        all_sites = sites.get("all",[])
        
        for j, site_coords in enumerate(all_sites):
            # Copy slab and add single_atom at this ontop site
            slab_copy = slab.copy()
            slab_copy.append(
                species=single_atom,
                coords=site_coords,
                coords_are_cartesian=True,
                validate_proximity=True
            )

            slab_copy.to(fmt="cif", filename = f"./{single_atom} Fairchem/{index}CoNi_{single_atom}_unrelaxed_{i}_Site_{j}.cif")
            slab_copy.to(fmt="poscar", filename = f"./{single_atom} Fairchem/{index}CoNi_{single_atom}_unrelaxed_{i}_Site_{j}.vasp")

def read_poscar(doc):
    with open(f"./{single_atom} Fairchem/{doc}", 'r') as f:
        lines = f.readlines()
    return lines

def read_z_coordinates(lines):
    z_coords = []
    for line in lines[8:]:
  
        z_coords.append(round(float(line.split()[2]),2))
    return  z_coords

def sort_by_z(z_coords):
    sorted_indices = sorted(range(len(z_coords)), key=lambda i: z_coords[i])
    sorted_z_coords = [z_coords[i] for i in sorted_indices]
    return sorted_z_coords


def write_fix_poscar(doc):
    lines = read_poscar(doc)
    z_coords = read_z_coordinates(lines)
    sorted_z_coords = sort_by_z(z_coords)
    unique_z_coords = sort_by_z(list(set(sorted_z_coords)))

    # Fix bottom layer of slab, change n to desired number of fixed layers
    n = 1
  
    with open(f'./{single_atom} Fairchem/{doc.split(".va")[0]}_fixed.vasp', 'w') as f:
        for line in lines[:7]:
            f.write(line)
        f.write('selective dynamics\n')
        f.write('Direct\n  ')

        for line in lines[8:]:
            z = round(float(line.split()[2]),2)
            temp = line.split()
            if z < unique_z_coords[n]:
                f.write(f'{temp[0]}   {temp[1]}   {temp[2]}   F F F\n')
            else:
                f.write(f'{temp[0]}   {temp[1]}   {temp[2]}   T T T\n')

all = os.listdir(f"./{single_atom} Fairchem/")
todo = []
for doc in all:
    if doc.endswith('.vasp'):
        todo.append(doc)
print(len(todo))
for i in todo:
    write_fix_poscar(i)

todo_2 = []
all = os.listdir(f"./{single_atom} Fairchem/")
for doc in all:
    if doc.endswith('_fixed.vasp'):
        todo_2.append(doc)
print(len(todo_2))
for i in todo_2:
    i_copy = Structure.from_file(f'./{single_atom} Fairchem/{i}').copy()
    ase_atoms = AseAtomsAdaptor.get_atoms(i_copy)
    ase_atoms.calc = calc
    dyn = BFGS(
        ase_atoms,
        trajectory=f"./{single_atom} Fairchem/{i.split('.va')[0]}_relaxed.traj"
    )
    dyn.run(fmax=0.05,steps=300)
    relaxed_structure = AseAtomsAdaptor.get_structure(ase_atoms)
    relaxed_structure.to(
        fmt="cif",
        filename=f"./{single_atom} Fairchem/{i.split('.va')[0]}_relaxed.cif"
    )
    relaxed_structure.to(
        fmt="poscar",
        filename=f"./{single_atom} Fairchem/{i.split('.va')[0]}_relaxed.vasp"
    )

    ase_atoms = AseAtomsAdaptor.get_atoms(i_copy)
    h_height = 1.5
            
    cell = ase_atoms.get_cell()
    normal = cell[2] / np.linalg.norm(cell[2])
            
    positions = ase_atoms.get_positions()
    symbols = ase_atoms.get_chemical_symbols()
    pt_indices = [idx for idx, sym in enumerate(symbols) if sym == single_atom]
    for idx_atom in pt_indices:
        pt_pos = positions[idx_atom]
        h_pos = pt_pos + normal * h_height
        ase_atoms.append(Atom("H", h_pos))
    ase_atoms.calc = calc
    dyn = BFGS(
        ase_atoms,
        trajectory=f"./{single_atom} Fairchem/{i.split('.va')[0]}_H_relaxed.traj"
    )
    dyn.run(fmax=0.05,steps=300)
    relaxed_structure = AseAtomsAdaptor.get_structure(ase_atoms)
    relaxed_structure.to(
        fmt="cif",
        filename=f"./{single_atom} Fairchem/{i.split('.va')[0]}_H_relaxed.cif"
    )
    relaxed_structure.to(
        fmt="poscar",
        filename=f"./{single_atom} Fairchem/{i.split('.va')[0]}_H_relaxed.vasp"
    )

print(f"{single_atom} predictions completed. Files saved in {single_atom} Fairchem folder. Run statistical analysis code next to screen results.")
