import pandas as pd
import qcelemental as qcel
import torch


#needed for tad to work properly
import tad_mctc as mctc
import tad_dftd3 as d3

#tad outputs energies in hartrees
h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")


df = pd.read_pickle("s66x8_cliff_elst_with_theta.pkl")

row_menh2_water = df.loc[df['system_id'] == "12_MeNH2-Water_1.00"]
row_water_dimer = df.loc[df['system_id'] == "01_Water-Water_1.00"]
print(row_water_dimer['dimer'].tolist()[0].to_string("psi4"))

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

def test_batch_two_uneven_dimers():


    R_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.geometry),
            torch.tensor(water_water_dimer.geometry)
        )
    )
    Z_AB = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.atomic_numbers),
            torch.tensor(water_water_dimer.atomic_numbers)
        )
    )

    R_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).geometry),
            torch.tensor(water_water_dimer.get_fragment(0).geometry),
        )
    )
    Z_A = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(0).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(0).atomic_numbers),
        )
    )
    

    R_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).geometry),
            torch.tensor(water_water_dimer.get_fragment(1).geometry),
        )
    )
    Z_B = mctc.batch.pack(
        (
            torch.tensor(menh2_water_dimer.get_fragment(1).atomic_numbers),
            torch.tensor(water_water_dimer.get_fragment(1).atomic_numbers),
        )
    )

    #Calculating the dftd3 dispersion energy via the supermolecular approach:
    energy_AB = d3.dftd3(Z_AB, R_AB, param,)
    energy_A = d3.dftd3(Z_A, R_A, param,)
    energy_B = d3.dftd3(Z_B, R_B, param,)
    total_AB = torch.sum(energy_AB, dim=-1)
    total_A = torch.sum(energy_A, dim=-1)
    total_B = torch.sum(energy_B, dim=-1)
    supermolecular_disp_E = total_AB - total_A - total_B
    print(supermolecular_disp_E)


if __name__ == "__main__":
    test_batch_two_uneven_dimers()
