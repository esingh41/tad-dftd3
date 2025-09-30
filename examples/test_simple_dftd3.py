from dftd3.interface import RationalDampingParam, DispersionModel
import tad_mctc as mctc
import qcelemental as qcel
import numpy as np
import torch

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
# res = water_water_dimer.get_dispersion(
#     sapt0_d3, 
#     grad=False
# )
#print(res.get("energy"))  # Results in atomic units

mask = mctc.batch.real_pairs(torch.tensor(Z_AB), mask_diagonal=True, mon_A_indices=torch.tensor(mon_A_indices), mon_B_indices=torch.tensor(mon_B_indices))
print(mask)

E_dimer = water_water_dimer.get_pairwise_dispersion(sapt0_d3)
E_mon_A = water_mon_A.get_pairwise_dispersion(sapt0_d3)
E_mon_B = water_mon_B.get_pairwise_dispersion(sapt0_d3)
print(np.array(E_dimer['additive pairwise energy'])[mask])
#print(np.array(E_mon_A['additive pairwise energy']))
#print(np.array(E_mon_B['additive pairwise energy']))

