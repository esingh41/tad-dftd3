import apnet_pt
import pandas as pd
import qcelemental as qcel
import torch
import numpy as np

#needed for tad to work properly
import tad_mctc as mctc
import tad_dftd3 as d3


#tad outputs energies in hartrees and wants bohr units
h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")

df = pd.read_pickle("s66x8_cliff_elst_with_theta.pkl")

row_menh2_water = df.loc[df['system_id'] == "12_MeNH2-Water_1.00"]
row_water_dimer = df.loc[df['system_id'] == "01_Water-Water_1.00"]
row_benzene_pyridine = df.loc[df['system_id'] == "27_Benzene-Pyridine_pi-pi_1.00"]
#print(row_benzene_pyridine['dimer'].tolist()[0].to_string("psi4"))

#fitted sapt0-d4 parameters for BJ damping function
param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
}


#taddftd3 needs units to be in bohr to work properly
menh2_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
N                    -1.008100800000    -0.528355160000     0.202192680000
H                    -1.188923790000    -2.359180490000     0.723479130000
H                    -2.121413420000    -0.313995840000    -1.337480310000
C                    -1.921680310000     1.112077560000     2.243810640000
H                    -1.724865800000     3.071847600000     1.662054110000
H                    -3.882890690000     0.784391550000     2.793966870000
H                    -0.727588730000     0.848110780000     3.893996460000
--
0 1
O                     4.023106790000     1.759569490000     0.398270440000
H                     2.478222790000     0.821750410000     0.071058880000
H                     5.122741160000     1.271079320000    -0.954293650000
units bohr
no_com
no_reorient
""")

water_water_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
O                    -1.326958230000    -0.105938530000     0.018788150000
H                    -1.931665240000     1.600174320000    -0.021710520000
H                     0.486644280000     0.079598090000     0.009862480000
--
0 1
O                     4.287563290000     0.049775580000     0.000960040000
H                     4.999275000000    -0.778642690000     1.448725300000
H                     4.991040900000    -0.850136520000    -1.407646550000
units bohr
no_com
no_reorient
""")

benzene_pyridine_dimer = qcel.models.Molecule.from_data("""
0 1
--
0 1
C                     1.547207570000     1.633049050000     0.355809190000
H                     2.770553190000     3.244031740000     0.651429050000
C                     2.587029620000    -0.737983290000    -0.126041300000
H                     4.616669650000    -0.967278880000    -0.208955410000
C                     1.009829210000    -2.812844480000    -0.513793930000
H                     1.815740040000    -4.651682210000    -0.898578930000
C                    -1.604595960000    -2.514294490000    -0.415544040000
H                    -2.829051200000    -4.123118380000    -0.717251620000
C                    -2.644644230000    -0.143676260000     0.076409500000
H                    -4.672700600000     0.084863400000     0.176466510000
C                    -1.068247630000     1.930172610000     0.457841350000
H                    -1.875660280000     3.767473860000     0.843305720000
--
0 1
N                    -4.484042050000     0.282741320000     6.476783420000
C                    -3.322076040000     2.470416420000     7.006028040000
H                    -4.545870340000     4.082676810000     7.313131390000
C                    -0.710068430000     2.736580070000     7.176426190000
H                     0.106007450000     4.557614870000     7.613361660000
C                     0.803510340000     0.625795100000     6.771620560000
H                     2.839820940000     0.758802100000     6.875242770000
C                    -0.369162810000    -1.658231910000     6.216684390000
H                     0.718893330000    -3.350711980000     5.868833170000
C                    -2.996853970000    -1.730113090000     6.093002910000
H                    -3.958446630000    -3.484635690000     5.660552930000
units bohr
no_com
no_reorient
""")

def test_batch_three_uneven_dimers_tad_super_vs_inter():

    #These are the SAPT0/adz dispersion energies in kcal/mol
    sapt_disp_energies = [
        -3.019950675708942, #water_menh2
        -1.585434260570913, #water_water
        -9.871413133865042, #benzene_pyridine
    ]

    R_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.geometry),
            torch.tensor(water_water_dimer.geometry),
            torch.tensor(benzene_pyridine_dimer.geometry),
        )
    )
    Z_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.atomic_numbers),
            torch.tensor(water_water_dimer.atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.atomic_numbers),
        )
    )

    R_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).geometry),
            torch.tensor(water_water_dimer.get_fragment(0).geometry),
            torch.tensor(benzene_pyridine_dimer.get_fragment(0).geometry),
        )
    )

    Z_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(0).atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.get_fragment(0).atomic_numbers),
        )
    )

    R_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).geometry),
            torch.tensor(water_water_dimer.get_fragment(1).geometry),
            torch.tensor(benzene_pyridine_dimer.get_fragment(1).geometry),
        )
    )
    Z_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(1).atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.get_fragment(1).atomic_numbers),
        )
    )

    #Calculating the dftd3 dispersion energy via the supermolecular approach:
    energy_AB = d3.dftd3(Z_AB, R_AB, param,)
    energy_A = d3.dftd3(Z_A, R_A, param,)
    energy_B = d3.dftd3(Z_B, R_B, param,)
    total_AB = torch.sum(energy_AB, dim=-1)
    total_A = torch.sum(energy_A, dim=-1)
    total_B = torch.sum(energy_B, dim=-1)
    supermolecular_disp_E = (total_AB - total_A - total_B) * h2kcalmol

    #assert np.allclose(supermolecular_disp_E, sapt_disp_energies, atol=0.1), f"{sapt_disp_energies = }, {supermolecular_disp_E.tolist() = }"

    #Calculating the dftd3 dispersion energy via the intermolecular approach
    mon_A_indices = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.fragments[0]),
            torch.tensor(water_water_dimer.fragments[0]),
            torch.tensor(benzene_pyridine_dimer.fragments[0]),
        )
    )

    mon_B_indices = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.fragments[1]),
            torch.tensor(water_water_dimer.fragments[1]),
            torch.tensor(benzene_pyridine_dimer.fragments[1]),
        )
    )
    intermolecular_disp_E = torch.sum(d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices), dim=-1)
    intermolecular_disp_E = intermolecular_disp_E * h2kcalmol
    print(f"{supermolecular_disp_E = }")
    print(f"{intermolecular_disp_E = }")
    print(f"{sapt_disp_energies = }")
    #assert np.allclose(intermolecular_disp_E, sapt_disp_energies, atol=0.1), f"{sapt_disp_energies = }, {intermolecular_disp_E.tolist() = }"
    assert np.allclose(intermolecular_disp_E, supermolecular_disp_E, atol=0.1), f"{supermolecular_disp_E.tolist() = }, {intermolecular_disp_E.tolist() = }"
    print(row_menh2_water['dftd3_intermolecular'].tolist()[0])

def test_batch_apnet_vs_tad_inter():

    #This is the batching for the tad_inter
    R_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.geometry),
            torch.tensor(water_water_dimer.geometry),
            torch.tensor(benzene_pyridine_dimer.geometry),
        )
    )
    Z_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.atomic_numbers),
            torch.tensor(water_water_dimer.atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.atomic_numbers),
        )
    )

    R_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).geometry),
            torch.tensor(water_water_dimer.get_fragment(0).geometry),
            torch.tensor(benzene_pyridine_dimer.get_fragment(0).geometry),
        )
    )

    Z_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(0).atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.get_fragment(0).atomic_numbers),
        )
    )

    R_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).geometry),
            torch.tensor(water_water_dimer.get_fragment(1).geometry),
            torch.tensor(benzene_pyridine_dimer.get_fragment(1).geometry),
        )
    )
    Z_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(1).atomic_numbers),
            torch.tensor(benzene_pyridine_dimer.get_fragment(1).atomic_numbers),
        )
    )

    #Calculating the dftd3 dispersion energy via the intermolecular approach
    mon_A_indices = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.fragments[0]),
            torch.tensor(water_water_dimer.fragments[0]),
            torch.tensor(benzene_pyridine_dimer.fragments[0]),
        )
    )

    mon_B_indices = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.fragments[1]),
            torch.tensor(water_water_dimer.fragments[1]),
            torch.tensor(benzene_pyridine_dimer.fragments[1]),
        )
    )
    tad_intermolecular_disp_E = torch.sum(d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices), dim=-1)
    tad_intermolecular_disp_E = tad_intermolecular_disp_E * h2kcalmol

    mols = [
        menh2_water_dimer,
        water_water_dimer,
        benzene_pyridine_dimer,
    ]

    batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
        [
            apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
                mol, r_cut=5.0, dimer_ind=n, r_cut_im=torch.inf
            )
            for n, mol in enumerate(mols)
        ]
    )

    ZA = batch.ZA
    ZB = batch.ZB

    RA = batch.RA * ang2bohr
    RB = batch.RB * ang2bohr

    print(batch.molecule_ind_A)

    #Separting the molecules out so I can merge with RB
    unique_values_A, repeats_A = np.unique(
            [batch.molecule_ind_A[i] for i in range(len(batch.molecule_ind_A))],
            return_counts=True,
    )
    RA_reshaped = []
    ZA_reshaped = []
    mon_A_indices = []
    idx = 0
    for repeat in repeats_A:
        RA_reshaped.append(RA[idx: idx + repeat])
        ZA_reshaped.append(ZA[idx: idx + repeat])
        idx += repeat

    unique_values_B, repeats_B = np.unique(
            [batch.molecule_ind_B[i] for i in range(len(batch.molecule_ind_B))],
            return_counts=True,
    )
    RB_reshaped = []
    ZB_reshaped = []
    idx = 0
    for repeat in repeats_B:
        RB_reshaped.append(RB[idx: idx + repeat])
        ZB_reshaped.append(ZB[idx: idx + repeat])
        idx += repeat
    
    mon_A_indices = []
    mon_B_indices = []
    for i, x in enumerate(unique_values_A):
        mon_A_indices.append(torch.arange(0, repeats_A[i]))
        mon_B_indices.append(torch.arange(repeats_A[i], repeats_A[i] + repeats_B[i]))
    R_AB = [torch.cat([a, b], dim=0) for a, b in zip(RA_reshaped, RB_reshaped)]
    Z_AB = [torch.cat([a, b], dim=0) for a, b in zip(ZA_reshaped, ZB_reshaped)]
    print(R_AB)

    R_AB = mctc.batch.pack(tuple(R_AB))
    Z_AB = mctc.batch.pack(tuple(Z_AB))
    mon_A_indices = mctc.batch.pack(tuple(mon_A_indices))
    mon_B_indices = mctc.batch.pack(tuple(mon_B_indices))

    apnet_intermolecular_disp_E = torch.sum(d3.dftd3(Z_AB, R_AB, param, mon_A_indices=mon_A_indices, mon_B_indices=mon_B_indices), dim=-1)
    apnet_intermolecular_disp_E = apnet_intermolecular_disp_E * h2kcalmol
    print(f"{apnet_intermolecular_disp_E = }")

    assert np.allclose(tad_intermolecular_disp_E, apnet_intermolecular_disp_E, atol=0.1), f"{tad_intermolecular_disp_E.tolist() = }, {apnet_intermolecular_disp_E.tolist() = }"



if __name__ == "__main__":
    test_batch_apnet_vs_tad_inter()
    #test_batch_three_uneven_dimers_tad_super_vs_inter()
    #print(row_benzene_pyridine['SAPT0_DISP_kcalmol'].tolist()[0])
