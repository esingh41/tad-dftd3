from dftd3.interface import RationalDampingParam, DispersionModel
import tad_mctc as mctc
import qcelemental as qcel
import numpy as np
import torch

h2kcalmol = qcel.constants.conversion_factor("hartree", "kcal/mol")
ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")

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

mols = [
        menh2_water_dimer,
        water_water_dimer,
        benzene_pyridine_dimer,
]


sapt0_d3 = RationalDampingParam(
        s8=0.738,
        a1=0.095,
        a2=3.637,
)

Z_AB = water_water_dimer.atomic_numbers
R_AB = water_water_dimer.geometry #takes in as Bohr

Z_A = water_water_dimer.get_fragment(0).atomic_numbers
R_A = water_water_dimer.get_fragment(0).geometry

Z_B = water_water_dimer.get_fragment(1).atomic_numbers
R_B = water_water_dimer.get_fragment(1).geometry

mon_A_indices = water_water_dimer.fragments[0]
mon_B_indices = water_water_dimer.fragments[1]
print(mon_B_indices)

water_water_dimer = DispersionModel(Z_AB, R_AB)
water_mon_A = DispersionModel(Z_A, R_A)
water_mon_B = DispersionModel(Z_B, R_B)
res = water_water_dimer.get_dispersion(
    sapt0_d3, 
    grad=False
)
#print(res.get("energy"))  # Results in atomic units


for x in mols:
        dimer = DispersionModel(x.atomic_numbers, x.geometry)
        mon_A = DispersionModel(x.get_fragment(0).atomic_numbers, x.get_fragment(0).geometry)
        mon_B = DispersionModel(x.get_fragment(1).atomic_numbers, x.get_fragment(1).geometry)

        e_AB = dimer.get_dispersion(
                sapt0_d3,
                grad=False
        )

        print(e_AB)
        
        e_A = mon_A.get_dispersion(
                sapt0_d3,
                grad=False
        )

        e_B = mon_B.get_dispersion(
                sapt0_d3,
                grad=False
        )

        print((e_AB['energy'] - e_A['energy'] - e_B['energy']) * h2kcalmol)



