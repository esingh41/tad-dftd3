import pandas as pd
import qcelemental
import tad_mctc as mctc
import torch
import apnet_pt
h2kcalmol = qcelemental.constants.conversion_factor("hartree", "kcal/mol")


import tad_dftd3 as d3


def water_dimer_test_case():
    filename="s66x8_cliff_elst_with_theta.pkl"
    df = pd.read_pickle("s66x8_cliff_elst_with_theta.pkl")
    row = df.loc[df['system_id'] == '01_Water-Water_1.00']
    dimer_qcel_mol = row['dimer'].tolist()[0]
    system_id = row['system_id'].tolist()[0]
    mon_A_indices = row['mon_A_indices'].tolist()[0]
    mon_B_indices = row['mon_B_indices'].tolist()[0]
    print(mon_A_indices)
    sapt_energy = row['SAPT0 DISP ENERGY adz'].tolist()[0] * h2kcalmol
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

    print(mon_A_indices)
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

def test_batching(filename, intermolecular=False):
    df = pd.read_pickle(filename)


    param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
    }

    R_AB = df['R_AB'].tolist()
    R_A = df['R_A'].tolist()
    R_B = df['R_B'].tolist()
    
    Z_AB = df['Z_AB'].tolist()
    Z_A = df['Z_A'].tolist()
    Z_B = df['Z_B'].tolist()

    R_AB = mctc.batch.pack(tuple(R_AB))
    R_A = mctc.batch.pack(tuple(R_A))
    R_B = mctc.batch.pack(tuple(R_B))

    Z_AB = mctc.batch.pack(tuple(Z_AB))
    Z_A = mctc.batch.pack(tuple(Z_A))
    Z_B = mctc.batch.pack(tuple(Z_B))

    torch.set_printoptions(threshold=10000)
    df['SAPT0_DISP_kcalmol'] =  df['SAPT0 DISP ENERGY adz'] * h2kcalmol

    if intermolecular: 
        mon_A_indices = df['mon_A_indices'].tolist()
        mon_B_indices = df['mon_B_indices'].tolist()

        mon_A_indices = mctc.batch.pack(tuple(mon_A_indices))
        mon_B_indices = mctc.batch.pack(tuple(mon_B_indices))
        energy_AB = d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices)
        print(energy_AB)
        total_AB = torch.sum(energy_AB, dim=-1)
        intermolecular_disp_E = total_AB
        df['dftd3_intermolecular'] = intermolecular_disp_E * h2kcalmol
        print(df[['system_id', 'dftd3_intermolecular', 'dftd3_supermolecular', 'SAPT0_DISP_kcalmol']].to_string())
        torch.set_printoptions(precision=10)

    else:
        mon_A_indices = None
        mon_B_indices = None
        energy_AB = d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices)
        energy_A = d3.dftd3(Z_A, R_A, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices)
        energy_B = d3.dftd3(Z_B, R_B, param, mon_A_indices=mon_B_indices, mon_B_indices=mon_B_indices)
        total_AB = torch.sum(energy_AB, dim=-1)
        total_A = torch.sum(energy_A, dim=-1)
        total_B = torch.sum(energy_B, dim=-1)
        supermolecular_disp_E = total_AB - total_A - total_B
        df['dftd3_supermolecular'] = supermolecular_disp_E * h2kcalmol
        print(df[['dftd3_supermolecular', 'SAPT0_DISP_kcalmol']].to_string())
        torch.set_printoptions(precision=10)
        #print("Actual  :", torch.sum(energy, dim=-1))
    df.to_pickle(filename)


def test_batching_two_water_dimers():
    df = pd.read_pickle("s66x8_cliff_elst_with_theta.pkl")
    water_dimers = df.loc[(df['system_id'] == '01_Water-Water_1.00') | (df['system_id'] == '01_Water-Water_2.00')]


    R_AB = water_dimers['R_AB'].tolist()
    R_A = water_dimers['R_A'].tolist()
    R_B = water_dimers['R_B'].tolist()
    
    Z_AB = water_dimers['Z_AB'].tolist()
    Z_A = water_dimers['Z_A'].tolist()
    Z_B = water_dimers['Z_B'].tolist()
    mon_A_indices = water_dimers['mon_A_indices'].tolist()
    #mon_A_indices = torch.stack(mon_A_indices, dim=0)
    mon_B_indices = water_dimers['mon_B_indices'].tolist()
   
    R_AB = mctc.batch.pack(tuple(R_AB))
    R_A = mctc.batch.pack(tuple(R_A))
    R_B = mctc.batch.pack(tuple(R_B))

    Z_AB = mctc.batch.pack(tuple(Z_AB))
    Z_A = mctc.batch.pack(tuple(Z_A))
    Z_B = mctc.batch.pack(tuple(Z_B))

    water_dimers['SAPT0_DISP_kcalmol'] =  df['SAPT0 DISP ENERGY adz'] * h2kcalmol

    mon_A_indices = mctc.batch.pack(tuple(mon_A_indices))
    #print(mon_A_indices)
    mon_B_indices = mctc.batch.pack(tuple(mon_B_indices))

    param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
    }

    energy_AB = d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices)
    print(energy_AB)
    total_AB = torch.sum(energy_AB, dim=-1)
    intermolecular_disp_E = total_AB
    water_dimers['dftd3_supermolecular'] = intermolecular_disp_E * h2kcalmol
    print(water_dimers[['dftd3_supermolecular', 'SAPT0_DISP_kcalmol']].to_string())
    torch.set_printoptions(precision=10)

def test_batching_two_dimers():
    df = pd.read_pickle("s66x8_cliff_elst_with_theta.pkl")
    df_water = df.loc[df['System Label'] == "01_Water-Water"]
    R_AB = df_water['R_AB'].tolist()[:1]
    Z_AB = df_water['Z_AB'].tolist()[:1]
    mon_A_indices = df_water['mon_A_indices'].tolist()[:1]
    mon_B_indices = df_water['mon_B_indices'].tolist()[:1]

    #print(R_AB)

    R_AB = mctc.batch.pack(tuple(R_AB))
    
    Z_AB = mctc.batch.pack(tuple(Z_AB))
    mon_A_indices = mctc.batch.pack(tuple(mon_A_indices))
    mon_B_indices = mctc.batch.pack(tuple(mon_B_indices))

    param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
    }

    energy = d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices)
    print(energy)
    total_energies = torch.sum(energy, dim=-1)

    return
    


def main():
    filename="s66x8_cliff_elst_with_theta.pkl"
    test_batching(filename, True)
    return
    test_batching_two_water_dimers()
    return
    water_dimer_test_case()
    return
    filename="s66x8_cliff_elst_with_theta.pkl"
    test_batching(filename)
    return
    df = pd.read_pickle(filename)
    R_AB = df['R_AB'].tolist()
    Z_AB = df['Z_AB'].tolist()
    mon_A_indices = df['mon_A_indices'].tolist()
    mon_B_indices = df['mon_B_indices'].tolist()
    test_batching(R_AB, Z_AB, mon_A_indices=None, mon_B_indices=None, filename=filename)
    return
    row_water = df.loc[df['system_id'] == '01_Water-Water_1.00']
    water_dimer = row_water['dimer'].tolist()[0]
    
    return
    df['R_AB'] = df['dimer'].apply(lambda x: torch.tensor(x.geometry))
    df['Z_AB'] = df['dimer'].apply(lambda x: torch.tensor(x.atomic_numbers))
    df['mon_A_indices'] = df['dimer'].apply(lambda x : torch.tensor(x.fragments[0]))
    df['mon_B_indices'] = df['dimer'].apply(lambda x : torch.tensor(x.fragments[1]))
    df.to_pickle("s66x8_cliff_elst_with_theta.pkl")
    return
    #df['R_dimer'] = df['coordinates'].apply(lambda x: torch.tensor(x))
    #df['Z_dimer'] = df['atomic_numbers'].apply(lambda x: torch.tensor(x))
    df['mon_A_indices'] = df['monAs'].apply(lambda x: torch.tensor(x))
    df['mon_B_indices'] = df['monBs'].apply(lambda x: torch.tensor(x))
    df.to_pickle("s66x8_cliff_elst_with_theta.pkl")
    return
    print(df['system_id'].tolist())
    row_water = df.loc[df['system_id'] == '01_Water-Water_1.00']
    d3_dispersion_dimer_qcel_mol(row_water)

if __name__ == "__main__":
    main()