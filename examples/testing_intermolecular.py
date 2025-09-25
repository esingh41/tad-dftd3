import pandas as pd
import qcelemental
import tad_mctc as mctc
import torch
h2kcalmol = qcelemental.constants.conversion_factor("hartree", "kcal/mol")


import tad_dftd3 as d3


def d3_dispersion_dimer_qcel_mol(row, dimer_qcel_mol=None):
    #dimer_qcel_mol = row['dimer']

    system_id = row['system_id'].tolist()[0]
    mon_A_indices = row['monAs'].tolist()[0]
    mon_B_indices = row['monBs'].tolist()[0]
    print(mon_A_indices)
    sapt_energy = row['SAPT0 DISP ENERGY adz'].tolist()[0]
    param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
    }
    Z_AB = dimer_qcel_mol.atomic_numbers
    Z_A = dimer_qcel_mol.get_fragment(0).atomic_numbers
    Z_B = dimer_qcel_mol.get_fragment(1).atomic_numbers

    R_AB = dimer_qcel_mol.geometry
    R_A = dimer_qcel_mol.get_fragment(0).geometry
    R_B = dimer_qcel_mol.get_fragment(1).geometry

    E_AB = torch.sum(d3.dftd3(torch.from_numpy(Z_AB), torch.from_numpy(R_AB), param))
    E_A = torch.sum(d3.dftd3(torch.from_numpy(Z_A), torch.from_numpy(R_A), param))
    E_B = torch.sum(d3.dftd3(torch.from_numpy(Z_B), torch.from_numpy(R_B), param))
    disp_E = E_AB - E_A - E_B
    disp_E = disp_E * h2kcalmol
    print(f"for {system_id}, the supermolecular disperion E is {disp_E}, and the SAPT0_adz energy is {sapt_energy} ")

    E_intermolecular_disp_E = torch.sum(d3.dftd3(torch.from_numpy(Z_AB), torch.from_numpy(R_AB), param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices))
    E_intermolecular_disp_E = E_intermolecular_disp_E * h2kcalmol
    print(f"for {system_id}, the intermolecular disperion E is {E_intermolecular_disp_E}, and the SAPT0_adz energy is {sapt_energy} ")
    return disp_E.item()

def merge_monomers(monA, monB):
    mol_str = f"""{monA.to_string("psi4").replace("units bohr", "").replace("no_com", "").replace("no_reorient", "")}--
    {monB.to_string("psi4").replace("units bohr", "").replace("no_com", "").replace("no_reorient", "")}units angstrom
no_com
no_reorient
    """
    # print(monA.to_string('psi4'))
    # print(monB.to_string('psi4'))
    print(mol_str)
    return qcelemental.models.Molecule.from_data(mol_str)

def main():
    df = pd.read_csv("s66x8.csv")
    #print(df.columns.tolist())
    #print(df['psi4_string_A'].tolist())
    print(df['system_id'].tolist())
    row_water = df.loc[df['system_id'] == '01_Water-Water_1.00']
    qcel_mol_A = qcelemental.models.Molecule.from_data(row_water['psi4_string_A'].tolist()[0], dtype='psi4')
    qcel_mol_B = qcelemental.models.Molecule.from_data(row_water['psi4_string_B'].tolist()[0], dtype='psi4')
    water_dimer = merge_monomers(qcel_mol_A, qcel_mol_B)

    print(water_dimer)
    d3_dispersion_dimer_qcel_mol(row_water, water_dimer)

if __name__ == "__main__":
    main()