
#include example implementation
from __future__ import annotations

from typing import Any

import torch
from tad_mctc import storch
from tad_mctc.autograd import is_functorch_tensor
from tad_mctc.batch import real_pairs
from tad_mctc.data import pse, radii
from tad_mctc.ncoord.count import exp_count 

#My version of the coordination numbers
from tad_mctc.ncoord.d3 import cn_d3_apnet
from . import data, defaults, model, ncoord
from .damping import dispersion_atm, rational_damping
from .reference import Reference
from .typing import (
    DD,
    CountingFunction,
    DampingFunction,
    Tensor,
    WeightingFunction,
)

__all__ = ["dftd3", "dispersion", "dispersion2", "dispersion3", "apnet_dispersion"]

#Borrowed from APNET
def get_distances(RA, RB, e_source, e_target):
        RA_source = RA.index_select(0, e_source)
        RB_target = RB.index_select(0, e_target)
        dR_xyz = RB_target - RA_source

        # Compute distances with safe operation for square root
        # dR = torch.sqrt(nn.functional.relu(torch.sum(dR_xyz**2, dim=-1)))
        dR = torch.sqrt(torch.sum(dR_xyz * dR_xyz, dim=-1).clamp_min(1e-10))
        return dR, dR_xyz

#Question, do I need to worry about implementing the derivatives or gradient
#Does the choice of counting function matter
def cn_d3(
    batch,
    *,
    counting_function: CountingFunction | None = None,
    rcov: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute the D3 fractional coordination (exponential counting function).

    Parameters
    ----------
    batch : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    counting_function : CountingFunction, optional
        Calculate weight for pairs. Defaults to
        :func:`tad_mctc.ncoord.count.exp_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to ``None``.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to ``None``.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function. For example, ``kcn``,
        the steepness of the counting function, which defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.
    
    Returns
    -------
    Tensor
        Coordination numbers for all atoms (shape: ``(..., nat)``).

    Raises
    ------
    ValueError
        If shape mismatch between ``numbers``, ``positions`` and
        ``rcov`` is detected.
    """
    RA = batch.RA

    #dictionary of defaults; RA is reference tensor, extracting device and precision
    #so can be used for other tensors
    dd: DD = {"device": RA.device, "dtype": RA.dtype}

    #What is this cutoff
    #I don't really have to care for a cutoff right, 
    #because I want to use all of the intramonomer edges anyways right
    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_CN_CUTOFF, **dd)

    if counting_function is None:
        counting_function = exp_count

    #I guess including the intermolecular edges would matter when you have
    #two monomers that are really close becasue then their sum of their covalent radii 
    #divided by their distance wouldn't be zero

    #If r_AB goes to infinity then the exponnet becomes zero
    #Need the covalent radii, because the sum of the covalent
    #radii divided the distance between the two atoms,
    #gives an indication of how "bonded" two atoms are.
    #They scale this ratio by 4/3, arbitrary much? They aren't doing that they are
    #dividing by inverse distance??? What is that?


    #But why not naively add all of ratios of sum of the covalent
    #radii divided by the distance between the two two atoms?
    #Why do you need to use an exponential function at all?

    #exponential form is to ensure it is differentiable?

        
    ############################################
    ##Getting the covalent radii for monomer A##
    ############################################
    ZA = batch.ZA
    #ZA =tensor([8, 1, 1])
    print(f"{ZA =}")
    rcov_A = radii.COV_D3(**dd)[ZA] 
    print(f"{rcov_A = }")
    #rcov_A = tensor([1.5874, 0.8063, 0.8063])
    e_AA_source = batch.e_AA_source
    e_AA_target = batch.e_AA_target
    rc_A = rcov_A.index_select(0, e_AA_source) + rcov_A.index_select(0, e_AA_target)
    print(f"{rc_A = }")
    #rc_A = tensor([2.3937, 2.3937, 2.3937, 1.6126, 2.3937, 1.6126])
    #rc_A contains the covalent radii sums


    ############################################
    ##Getting the covalent radii for monomer B##
    ############################################
    ZB = batch.ZB
    rcov_B = radii.COV_D3(**dd)[ZB] 
    print(f"{rcov_B = }")
    #rcov_B = tensor([1.5874, 0.8063, 0.8063])
    e_BB_source = batch.e_BB_source
    e_BB_target = batch.e_BB_target
    rc_B = rcov_B.index_select(0, e_BB_source) + rcov_B.index_select(0, e_BB_target)
    print(f"{rc_B = }")
    #rc_B = tensor([2.3937, 2.3937, 2.3937, 1.6126, 2.3937, 1.6126])
    ############################################
    #Getting the coordination #s for monomer A##
    ############################################
    RA = batch.RA
    e_AA_source = batch.e_AA_source
    e_AA_target = batch.e_AA_target
    dRA, _ = get_distances(RA, RA, e_AA_source, e_AA_target)
    print(f"{dRA = }")
    #dRA = tensor([0.9581, 0.9647, 0.9581, 1.5118, 0.9647, 1.5118]) dRA is 1D, so covalent radii also need to be one D
    
    cf_A = torch.where(
        (dRA <= cutoff),
        counting_function(dRA, rc_A),
        torch.tensor(0.0, **dd)
    )
    print(f"{cf_A = }")
    #cf_A = tensor([1.0000, 1.0000, 1.0000, 0.7440, 1.0000, 0.7440])
    #Hmmm, what does a coordination number of 0.74 mean? 3/4s of a bond?
    #Oxygen has the same coordination number with respect to both Hs makes sense
    #cf_A = scatter_sum_compile(cf_A, e_AA_source, 1,)
    cn_A_size = e_AA_source.max().item() + 1
    cn_A = torch.zeros(cn_A_size, dtype=cf_A.dtype)
    cn_A.scatter_reduce_(0, e_AA_source, cf_A, reduce="sum", include_self=False)
    print(f"{cn_A = }")
    #Makes sense oxygen has two bonding partners, and then hydrogen also has close to two? Weird
    #cn_A = tensor([2.0000, 1.7440, 1.7440])
    
    #Computing the coordination numbers for Monomer B
    RB=batch.RB
    e_BB_source = batch.e_BB_source
    e_BB_target = batch.e_BB_target
    dRB, _ = get_distances(RB, RB, e_BB_source, e_BB_target)
    cf_B = torch.where(
        (dRB <= cutoff),
        counting_function(dRB, rc_B),
        torch.tensor(0.0, **dd)
    )

    cn_B_size = e_BB_source.max().item() + 1
    cn_B = torch.zeros(cn_B_size, dtype=cf_B.dtype)
    cn_B.scatter_reduce_(0, e_BB_source, cf_B, reduce="sum", include_self=False)
    print(f"{cn_B = }")
    #Why are the numbers different
    #cn_B = tensor([2.0000, 1.7435, 1.7435])
    return cn_A, cn_B

def apnet_dispersion(
    batch,
    param: dict[str, Tensor],
    *,
    ref: Reference | None = None,
    rcov: Tensor | None = None,
    rvdw: Tensor | None = None,
    r4r2: Tensor | None = None,
    cutoff: Tensor | None = None,
    counting_function: CountingFunction = ncoord.exp_count,
    weighting_function: WeightingFunction = model.gaussian_weight,
    damping_function: DampingFunction = rational_damping,
    pairwise_matrix=False,
    chunk_size: int | None = None,
    **kwargs,

):
    """

    ref --> This includes the C6 and the CNS for each element based on a set of reference
    molecules which represent different coordination environments. For C, they might use
    Ethye (CN=2), Ethene(CN=3), Ethane(CN=4), Free atom. The C6 and CNs are computed 
    for each atom in the molecules. Then we compute the CN for our atom A and CN for atom B, 
    and that is used to determine the C6. Each computed C_AB6 is tagged with the CNs of 
    #atoms A and B in that reference molecule. These are supporting points for interpolation
    #Then using the CNs of atoms A and B in the molecule, the method performs a 2D 
    #interpolation

    #They use a gaussian weighted average (L). So the weight is just plugging 
    #in the CN for the atom, and then its difference from the reference
    """
    RA = batch.RA
    dd: DD = {"device": RA.device, "dtype": RA.dtype}

    if cutoff is None:
        cutoff = torch.tensor(defaults.D3_DISP_CUTOFF, **dd)
    if ref is None:
        ref = Reference(**dd)
    # if rcov is None:
    #     rcov = radii.COV_D3(**dd)[numbers]
    # if rvdw is None:
    #     rvdw = radii.VDW_PAIRWISE(**dd)[
    #         numbers.unsqueeze(-1), numbers.unsqueeze(-2)
    #     ]
    # if r4r2 is None:
    #     r4r2 = data.R4R2(**dd)[numbers]

    # if numbers.shape != positions.shape[:-1]:
    #     raise ValueError(
    #         "Shape of positions is not consistent with atomic numbers.",
    #     )
    # if numbers.shape != r4r2.shape:
    #     raise ValueError(
    #         "Shape of expectation values is not consistent with atomic numbers.",
    #     )

    # if not is_functorch_tensor(numbers):
    #     if torch.max(numbers) >= defaults.MAX_ELEMENT:
    #         raise ValueError(
    #             f"No D3 parameters available for Z > {defaults.MAX_ELEMENT-1} "
    #             f"({pse.Z2S[defaults.MAX_ELEMENT]})."
    #         )
    

    #fractional coordination numbers computation encodes system dependent information
    #Oh I see fractional coordination numbers used to compute weights, the weights ar then used to compute
    #c6. That's how c6 coeffcients encode environmental information

    #Contribution of an atom to the total disperson coeffcient of a molecule
    #is dependent on its chemical environmnet. Fractionally occupied atomic orbitals
    #become doubly occupied, energetically lower-lying molecule orbitals
    #This increases electronic excitation energies, becuase the electrons are now 
    #pushed to a lower energy state so increasing it to that higher energy state
    #Thus, the resulting atomic polarizabilities and derived C6 coefficients
    #are much smaller in molecules than in free atoms.
    #C6 coefficients are much smaller in molecules then in free atoms
    #due to quenching of atomic states, is C6 divided by CN?


    cn_A, cn_B = cn_d3(
        batch, 
    )
    
    ZA = batch.ZA
    RA = batch.RA

    ZB = batch.ZB
    RB = batch.RB
    
    weights_A = model.weight_references(ZA, cn_A, ref, weighting_function)
    weights_B = model.weight_references(ZB, cn_B, ref, weighting_function)

    #ASK MENTOR ABOUT CHUNK SIZE
    c6 = model._atomic_c6_full_apnet(
         ZA=ZA, 
         ZB=ZB,
         weights_A=weights_A,
         weights_B=weights_B,
         reference=ref
         
    )
    #model.atomic_c6 calls on _atomic_c6_full or _atomic_c6_chunked if the chunksize is none
    #I think dealing when the chunk_size is greater than zero is above my paygrade.

    c6 = model.atomic_c6(numbers, weights, ref, chunk_size=chunk_size)

    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )


    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8_old = c6 * qq
    print(f"{c6 = }")
    print(f"{c8_old = }")
    #Computing the c8 coefficient with multipole expectations, well that's not going to 
    #work unless I compute c6 correctly
    ZA = ZA.index_select(0, e_source)
    ZB = ZB.index_select(0, e_target)
    r4r2_A = data.R4R2(**dd)[ZA]
    r4r2_B = data.R4R2(**dd)[ZB] 
    print(r4r2_A)
    qAqB = 3 * r4r2_A.unsqueeze(-1) * r4r2_B.unsqueeze(-2)
    c8_new = c6 * qAqB

    print(f"{c8_new = }")
    return


    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, qq, param, **kwargs),
        torch.tensor(0.0, **dd),
    )

    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, qq, param, **kwargs),
        torch.tensor(0.0, **dd),
    )

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))

    #Not multiplying by 0.5 here because I adjusted the mask so it only returns AB
    #interactions, and no BA interactions so no double counting so no need to multiply by 0.5
    if pairwise_matrix and mon_A_indices is not None and mon_B_indices is not None:
        e6 = -1 * (c6 * t6) * s6
        e8 = -1 * (c8 * t8) * s8
        return e6 + e8, mask
    
    return s6 * e6 + s8 * e8
