"""Common absorbing-mask path used by atom and bond coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vadgm.chemistry import BondToken


class LinearRevealSchedule:
    """The shared schedule kappa(t)=t and its hazard reparameterization."""

    @staticmethod
    def kappa(t: torch.Tensor) -> torch.Tensor:
        if torch.any((t < 0) | (t > 1)):
            raise ValueError("Diffusion time must lie in [0, 1]")
        return t

    @staticmethod
    def hazard(t: torch.Tensor) -> torch.Tensor:
        if torch.any((t < 0) | (t >= 1)):
            raise ValueError("Finite hazard requires diffusion time in [0, 1)")
        return -torch.log1p(-t)

    @staticmethod
    def inverse_hazard(s: torch.Tensor) -> torch.Tensor:
        if torch.any(s < 0):
            raise ValueError("Hazard time must be non-negative")
        return -torch.expm1(-s)

    def conditional_reveal_probability(
        self,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ) -> torch.Tensor:
        if torch.any(t_end < t_start):
            raise ValueError("t_end must not precede t_start")
        start = self.kappa(t_start)
        end = self.kappa(t_end)
        denominator = 1.0 - start
        probability = torch.where(
            denominator > 0,
            (end - start) / denominator,
            torch.ones_like(denominator),
        )
        return probability.clamp(0.0, 1.0)


@dataclass(frozen=True)
class CorruptedBatch:
    atom_ids: torch.Tensor
    bond_ids: torch.Tensor
    atom_revealed: torch.Tensor
    bond_revealed: torch.Tensor
    time: torch.Tensor


def sample_training_times(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if not 0 <= epsilon < 0.5:
        raise ValueError("epsilon must lie in [0, 0.5)")
    values = torch.rand(batch_size, device=device, generator=generator)
    return epsilon + (1.0 - 2.0 * epsilon) * values


def corrupt_batch(
    clean_atom_ids: torch.Tensor,
    clean_bond_ids: torch.Tensor,
    atom_valid_mask: torch.Tensor,
    bond_valid_mask: torch.Tensor,
    time: torch.Tensor,
    *,
    atom_mask_id: int,
    atom_pad_id: int,
    schedule: LinearRevealSchedule | None = None,
    generator: torch.Generator | None = None,
) -> CorruptedBatch:
    """Independently reveal coordinates with the same class-independent kappa."""

    if clean_atom_ids.ndim != 2 or clean_bond_ids.ndim != 2:
        raise ValueError("Expected batched atom and upper-triangle bond tensors")
    if time.shape != (clean_atom_ids.shape[0],):
        raise ValueError("time must contain one value per graph")
    if atom_valid_mask.shape != clean_atom_ids.shape:
        raise ValueError("atom_valid_mask shape mismatch")
    if bond_valid_mask.shape != clean_bond_ids.shape:
        raise ValueError("bond_valid_mask shape mismatch")

    reveal_schedule = schedule or LinearRevealSchedule()
    probability = reveal_schedule.kappa(time)
    atom_random = torch.rand(
        clean_atom_ids.shape,
        device=clean_atom_ids.device,
        generator=generator,
    )
    bond_random = torch.rand(
        clean_bond_ids.shape,
        device=clean_bond_ids.device,
        generator=generator,
    )
    atom_revealed = atom_valid_mask & (atom_random < probability[:, None])
    bond_revealed = bond_valid_mask & (bond_random < probability[:, None])

    atom_ids = torch.full_like(clean_atom_ids, atom_pad_id)
    atom_ids[atom_valid_mask] = atom_mask_id
    atom_ids[atom_revealed] = clean_atom_ids[atom_revealed]
    bond_ids = torch.full_like(clean_bond_ids, BondToken.PAD)
    bond_ids[bond_valid_mask] = BondToken.MASK
    bond_ids[bond_revealed] = clean_bond_ids[bond_revealed]
    return CorruptedBatch(atom_ids, bond_ids, atom_revealed, bond_revealed, time)

