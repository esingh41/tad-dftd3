#from tad_mctc.ncoord.d3 import cn_d3_apnet
from tad_dftd3.apnet_dispersion_batch import cn_d3_batch, apnet_dispersion_batch
import qcelemental
import apnet_pt
import torch

param = {
    "a1": torch.tensor(0.095),
    "s8": torch.tensor(0.738),
    "a2": torch.tensor(3.637),
}


water_water_dimer = qcelemental.models.Molecule.from_data("""
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

mols = [
    water_water_dimer
]

batch = apnet_pt.pt_datasets.ap2_fused_ds.ap2_fused_collate_update_no_target(
    [
        apnet_pt.pt_datasets.ap2_fused_ds.qcel_dimer_to_fused_data(
            mol, r_cut=5.0, dimer_ind=n, r_cut_im=torch.inf
        )
        for n, mol in enumerate(mols)
    ]
)

def test_cn():
    #cn_A, cn_B = cn_d3_batch(batch)
    apnet_dispersion_batch(batch, param)


if __name__ == "__main__":
    test_cn()
"""
Intel core architecture is the base of most architecture.
Why are we limited to fetching one instruction at a time. Why can't we do more.
The Intel core has two alus, that's alot of them. Maybe that's so you can do different ALU
ops at the same time. 

Why is cache important. Cache is close to processor so you don't need
You can have a lot of caches. The chips get more expensive 
The OS deals with cache. If you keep asking for the same peiece of data 
out of memory, the OS makes a wild guess that maybe it should keep the data
close to you. 

The Intel core makes a distinction with memory for instructions and memory
for data. The LC3 is Von Neumann

There are times we don't change instructions as much as we do data. 

In LC3, we don't have an instruction that can directly load, add, and then store
back into memory. 

In Intel, the complex instruction sinto micro ops. So a single instruction
in the old instruction set is changed into the smaller instructions automatically.
So all of your old software runs on the old instruction set, but under the hood
we translate to micro ops

But what happens if one instruction needs the value at register 3, and 
another instruction is storing a value at register 3. Can't do that

But what if we just reorder the instruction and runs the instructions
that don't depend on each other.

core often refers to multiple ececution system. 
"""