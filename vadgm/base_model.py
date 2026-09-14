"""Permutation-equivariant absorbing-mask base molecular graph generator."""

from __future__ import annotations

import os
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import functional as F

from vadgm.chemistry import AtomVocabulary, BondToken, TokenGraph
from vadgm.diffusion import LinearRevealSchedule, corrupt_batch, sample_training_times


@dataclass(frozen=True)
class BaseModelConfig:
    atom_vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.1
    max_atoms: int = 38


@dataclass(frozen=True)
class BaseLoss:
    total: torch.Tensor
    atom: torch.Tensor
    bond: torch.Tensor
    masked_atoms: int
    masked_bonds: int


@dataclass(frozen=True)
class BaseSampleDiagnostics:
    steps: int
    model_forwards: int
    revealed_atoms: int
    revealed_bonds: int


def upper_bonds_to_dense(
    bond_ids: torch.Tensor,
    n_atoms: torch.Tensor,
    *,
    max_atoms: int,
    diagonal_value: int = BondToken.PAD,
) -> torch.Tensor:
    """Expand local upper triangles without assuming a shared padded pair order."""

    batch_size = bond_ids.shape[0]
    dense = torch.full(
        (batch_size, max_atoms, max_atoms),
        diagonal_value,
        dtype=bond_ids.dtype,
        device=bond_ids.device,
    )
    output_length = bond_ids.shape[1]
    if output_length == 0:
        return dense
    rows, columns, valid = _packed_pair_indices(
        n_atoms,
        max_atoms=max_atoms,
        output_length=output_length,
    )
    batch = torch.arange(batch_size, device=bond_ids.device)[:, None].expand_as(rows)
    dense[batch[valid], rows[valid], columns[valid]] = bond_ids[valid]
    dense[batch[valid], columns[valid], rows[valid]] = bond_ids[valid]
    return dense


def dense_bonds_to_upper(
    dense: torch.Tensor,
    n_atoms: torch.Tensor,
    *,
    output_length: int,
    pad_value: int = BondToken.PAD,
) -> torch.Tensor:
    batch_size = dense.shape[0]
    upper = torch.full(
        (batch_size, output_length, *dense.shape[3:]),
        pad_value,
        dtype=dense.dtype,
        device=dense.device,
    )
    if output_length == 0:
        return upper
    rows, columns, valid = _packed_pair_indices(
        n_atoms,
        max_atoms=dense.shape[1],
        output_length=output_length,
    )
    batch = torch.arange(batch_size, device=dense.device)[:, None].expand_as(rows)
    gathered = dense[batch, rows, columns]
    upper[valid] = gathered[valid]
    return upper


@lru_cache(maxsize=32)
def _packed_pair_lookup(
    max_atoms: int,
    output_length: int,
    device_type: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache local packed-edge endpoints once per shape and device."""

    rows = torch.zeros((max_atoms + 1, output_length), dtype=torch.long)
    columns = torch.zeros_like(rows)
    for count in range(2, max_atoms + 1):
        pairs = torch.triu_indices(count, count, offset=1)
        edge_count = min(pairs.shape[1], output_length)
        rows[count, :edge_count] = pairs[0, :edge_count]
        columns[count, :edge_count] = pairs[1, :edge_count]
    device = torch.device(device_type, device_index)
    return rows.to(device), columns.to(device)


def _packed_pair_indices(
    n_atoms: torch.Tensor,
    *,
    max_atoms: int,
    output_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_atoms.ndim != 1 or torch.any(n_atoms < 0) or torch.any(n_atoms > max_atoms):
        raise ValueError("n_atoms is incompatible with max_atoms")
    required = n_atoms * (n_atoms - 1) // 2
    if torch.any(required > output_length):
        raise ValueError("Bond tensor is too short for n_atoms")
    lookup_rows, lookup_columns = _packed_pair_lookup(
        max_atoms,
        output_length,
        n_atoms.device.type,
        n_atoms.device.index,
    )
    rows = lookup_rows[n_atoms]
    columns = lookup_columns[n_atoms]
    valid = torch.arange(output_length, device=n_atoms.device)[None] < required[:, None]
    return rows, columns, valid


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        half = max(hidden_dim // 2, 1)
        frequencies = torch.exp(
            torch.linspace(0.0, -9.0, half, dtype=torch.float32)
        )
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Sequential(
            nn.Linear(2 * half, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phase = time[:, None] * self.frequencies[None, :] * (2.0 * torch.pi)
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.projection(features)


class EquivariantGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = nodes.shape[1]
        receiver = nodes[:, :, None, :].expand(-1, -1, count, -1)
        sender = nodes[:, None, :, :].expand(-1, count, -1, -1)
        messages = self.message(torch.cat((receiver, sender, edges), dim=-1))
        messages = messages * pair_mask[..., None]
        degree = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=2) / degree
        node_delta = self.node_update(torch.cat((nodes, aggregate), dim=-1))
        nodes = self.node_norm(nodes + node_delta)
        nodes = nodes * node_mask[..., None]

        left = nodes[:, :, None, :].expand(-1, -1, count, -1)
        right = nodes[:, None, :, :].expand(-1, count, -1, -1)
        symmetric_features = torch.cat(
            (left + right, torch.abs(left - right), edges),
            dim=-1,
        )
        edge_delta = self.edge_update(symmetric_features)
        edges = self.edge_norm(edges + edge_delta)
        edges = edges * pair_mask[..., None]
        return nodes, edges


class BaseGenerator(nn.Module):
    """Predict clean atom and bond categories for every graph coordinate."""

    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.atom_embedding = nn.Embedding(config.atom_vocab_size + 2, hidden)
        self.bond_embedding = nn.Embedding(BondToken.CLEAN_SIZE + 2, hidden)
        self.time_embedding = TimeEmbedding(hidden)
        self.size_embedding = nn.Embedding(config.max_atoms + 1, hidden)
        self.layers = nn.ModuleList(
            EquivariantGraphLayer(hidden, config.dropout)
            for _ in range(config.num_layers)
        )
        self.atom_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.atom_vocab_size),
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, BondToken.CLEAN_SIZE),
        )

    def forward(
        self,
        atom_ids: torch.Tensor,
        bond_ids: torch.Tensor,
        atom_valid_mask: torch.Tensor,
        bond_valid_mask: torch.Tensor,
        time: torch.Tensor,
        n_atoms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_atoms = atom_ids.shape
        if max_atoms > self.config.max_atoms:
            raise ValueError(
                f"Batch has {max_atoms} atoms, model maximum is {self.config.max_atoms}"
            )
        if time.shape != (batch_size,) or n_atoms.shape != (batch_size,):
            raise ValueError("time and n_atoms must contain one value per graph")
        if torch.any(n_atoms > self.config.max_atoms):
            raise ValueError("n_atoms exceeds the configured maximum")

        dense_bonds = upper_bonds_to_dense(
            bond_ids,
            n_atoms,
            max_atoms=max_atoms,
        )
        node_mask = atom_valid_mask.bool()
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        diagonal = torch.eye(max_atoms, dtype=torch.bool, device=atom_ids.device)
        pair_mask = pair_mask & ~diagonal[None, :, :]

        condition = self.time_embedding(time) + self.size_embedding(n_atoms)
        nodes = self.atom_embedding(atom_ids) + condition[:, None, :]
        edges = self.bond_embedding(dense_bonds) + condition[:, None, None, :]
        nodes = nodes * node_mask[..., None]
        edges = edges * pair_mask[..., None]
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, node_mask, pair_mask)

        atom_logits = self.atom_head(nodes)
        dense_bond_logits = self.bond_head(edges)
        bond_logits = dense_bonds_to_upper(
            dense_bond_logits,
            n_atoms,
            output_length=bond_ids.shape[1],
            pad_value=0,
        )
        atom_logits = atom_logits.masked_fill(~node_mask[..., None], 0.0)
        bond_logits = bond_logits.masked_fill(~bond_valid_mask[..., None], 0.0)
        return atom_logits, bond_logits

    def posterior(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        atom_logits, bond_logits = self.forward(*args)
        return atom_logits.softmax(dim=-1), bond_logits.softmax(dim=-1)


def base_denoising_loss(
    atom_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    clean_atom_ids: torch.Tensor,
    clean_bond_ids: torch.Tensor,
    partial_atom_ids: torch.Tensor,
    partial_bond_ids: torch.Tensor,
    atom_valid_mask: torch.Tensor,
    bond_valid_mask: torch.Tensor,
    *,
    atom_mask_id: int,
    edge_loss_weight: float = 1.0,
) -> BaseLoss:
    atom_supervision = atom_valid_mask & (partial_atom_ids == atom_mask_id)
    bond_supervision = bond_valid_mask & (partial_bond_ids == BondToken.MASK)
    safe_atom_targets = torch.where(atom_valid_mask, clean_atom_ids, 0)
    safe_bond_targets = torch.where(bond_valid_mask, clean_bond_ids, 0)
    atom_values = F.cross_entropy(
        atom_logits.transpose(1, 2),
        safe_atom_targets,
        reduction="none",
    )
    if bond_logits.shape[1] == 0:
        bond_values = torch.zeros_like(safe_bond_targets, dtype=atom_logits.dtype)
    else:
        bond_values = F.cross_entropy(
            bond_logits.transpose(1, 2),
            safe_bond_targets,
            reduction="none",
        )
    atom_count = atom_supervision.sum(dim=1)
    bond_count = bond_supervision.sum(dim=1)
    atom_per_graph = (atom_values * atom_supervision).sum(dim=1) / atom_count.clamp_min(1)
    bond_per_graph = (bond_values * bond_supervision).sum(dim=1) / bond_count.clamp_min(1)
    atom_loss = atom_per_graph.mean()
    bond_loss = bond_per_graph.mean()
    total = atom_loss + edge_loss_weight * bond_loss
    return BaseLoss(
        total=total,
        atom=atom_loss,
        bond=bond_loss,
        masked_atoms=int(atom_supervision.sum().item()),
        masked_bonds=int(bond_supervision.sum().item()),
    )


def make_corrupted_training_batch(
    batch: Mapping[str, object],
    vocabulary: AtomVocabulary,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    tensor_batch = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    clean_atoms = tensor_batch["atom_ids"]
    clean_bonds = tensor_batch["bond_ids"]
    time = sample_training_times(
        clean_atoms.shape[0],
        device=clean_atoms.device,
        generator=generator,
    )
    corrupted = corrupt_batch(
        clean_atoms,
        clean_bonds,
        tensor_batch["atom_mask"],
        tensor_batch["bond_mask"],
        time,
        atom_mask_id=vocabulary.mask_id,
        atom_pad_id=vocabulary.pad_id,
        generator=generator,
    )
    model_inputs = {
        "atom_ids": corrupted.atom_ids,
        "bond_ids": corrupted.bond_ids,
        "atom_valid_mask": tensor_batch["atom_mask"],
        "bond_valid_mask": tensor_batch["bond_mask"],
        "time": time,
        "n_atoms": tensor_batch["n_atoms"],
    }
    return model_inputs, clean_atoms, clean_bonds


def train_base_epoch(
    model: BaseGenerator,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    vocabulary: AtomVocabulary,
    *,
    device: torch.device | str,
    edge_loss_weight: float = 1.0,
    generator: torch.Generator | None = None,
    progress_desc: str | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    samples = 0
    batches_iterable = (
        tqdm(loader, desc=progress_desc, unit="batch", leave=False)
        if progress_desc
        else loader
    )
    for batch in batches_iterable:
        model_inputs, clean_atoms, clean_bonds = make_corrupted_training_batch(
            batch,
            vocabulary,
            device=device,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            atom_logits, bond_logits = model(**model_inputs)
            losses = base_denoising_loss(
                atom_logits,
                bond_logits,
                clean_atoms,
                clean_bonds,
                model_inputs["atom_ids"],
                model_inputs["bond_ids"],
                model_inputs["atom_valid_mask"],
                model_inputs["bond_valid_mask"],
                atom_mask_id=vocabulary.mask_id,
                edge_loss_weight=edge_loss_weight,
            )
        if grad_scaler is not None:
            grad_scaler.scale(losses.total).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            losses.total.backward()
            optimizer.step()
        batch_size = clean_atoms.shape[0]
        totals += torch.stack(
            (losses.total.detach(), losses.atom.detach(), losses.bond.detach())
        ).to(dtype=torch.float64) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Training loader produced no batches")
    means = (totals / samples).cpu().tolist()
    return {
        "loss": means[0],
        "atom_loss": means[1],
        "bond_loss": means[2],
    }


@torch.no_grad()
def evaluate_base_epoch(
    model: BaseGenerator,
    loader: Iterable[Mapping[str, object]],
    vocabulary: AtomVocabulary,
    *,
    device: torch.device | str,
    edge_loss_weight: float = 1.0,
    generator: torch.Generator | None = None,
    progress_desc: str | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    samples = 0
    batches_iterable = (
        tqdm(loader, desc=progress_desc, unit="batch", leave=False)
        if progress_desc
        else loader
    )
    for batch in batches_iterable:
        model_inputs, clean_atoms, clean_bonds = make_corrupted_training_batch(
            batch,
            vocabulary,
            device=device,
            generator=generator,
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            atom_logits, bond_logits = model(**model_inputs)
            losses = base_denoising_loss(
                atom_logits,
                bond_logits,
                clean_atoms,
                clean_bonds,
                model_inputs["atom_ids"],
                model_inputs["bond_ids"],
                model_inputs["atom_valid_mask"],
                model_inputs["bond_valid_mask"],
                atom_mask_id=vocabulary.mask_id,
                edge_loss_weight=edge_loss_weight,
            )
        batch_size = clean_atoms.shape[0]
        totals += torch.stack((losses.total, losses.atom, losses.bond)).to(
            dtype=torch.float64
        ) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("Validation loader produced no batches")
    means = (totals / samples).cpu().tolist()
    return dict(zip(("loss", "atom_loss", "bond_loss"), means))


def save_base_checkpoint(
    path: str | Path,
    model: BaseGenerator,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    best_validation_loss: float,
    metadata: Mapping[str, object] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "torch_rng_state": torch.get_rng_state(),
        "metadata": dict(metadata or {}),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary = target.with_suffix(target.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_base_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
) -> tuple[BaseGenerator, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = BaseGenerator(BaseModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        # ``map_location='cuda'`` also maps the saved CPU RNG tensor to CUDA,
        # but PyTorch's default-generator API requires a CPU ByteTensor.
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]]
            )
    return model, payload


def _sample_categories(
    logits: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    flat = logits.reshape(-1, logits.shape[-1])
    probabilities = flat.softmax(dim=-1)
    sampled = torch.multinomial(probabilities, 1, generator=generator)
    return sampled.reshape(logits.shape[:-1])


@torch.no_grad()
def sample_base_graphs(
    model: BaseGenerator,
    vocabulary: AtomVocabulary,
    n_atoms: Sequence[int],
    *,
    steps: int,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    progress_desc: str | None = None,
) -> tuple[list[TokenGraph], BaseSampleDiagnostics]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not n_atoms or min(n_atoms) <= 0:
        raise ValueError("n_atoms must contain positive graph sizes")
    max_atoms = max(n_atoms)
    if max_atoms > model.config.max_atoms:
        raise ValueError("Requested graph exceeds model max_atoms")
    batch_size = len(n_atoms)
    max_bonds = max_atoms * (max_atoms - 1) // 2
    n_tensor = torch.tensor(n_atoms, dtype=torch.long, device=device)
    atom_valid = torch.arange(max_atoms, device=device)[None, :] < n_tensor[:, None]
    edge_counts = n_tensor * (n_tensor - 1) // 2
    bond_valid = torch.arange(max_bonds, device=device)[None, :] < edge_counts[:, None]
    atom_ids = torch.full(
        (batch_size, max_atoms),
        vocabulary.pad_id,
        dtype=torch.long,
        device=device,
    )
    atom_ids[atom_valid] = vocabulary.mask_id
    bond_ids = torch.full(
        (batch_size, max_bonds),
        BondToken.PAD,
        dtype=torch.long,
        device=device,
    )
    bond_ids[bond_valid] = BondToken.MASK

    model.eval()
    schedule = LinearRevealSchedule()
    model_forwards = 0
    step_iterable = (
        tqdm(range(steps), desc=progress_desc, unit="step", leave=False)
        if progress_desc
        else range(steps)
    )
    for step in step_iterable:
        start = torch.full(
            (batch_size,),
            step / steps,
            dtype=torch.float32,
            device=device,
        )
        end = torch.full_like(start, (step + 1) / steps)
        atom_logits, bond_logits = model(
            atom_ids,
            bond_ids,
            atom_valid,
            bond_valid,
            start,
            n_tensor,
        )
        model_forwards += 1
        atom_candidates = _sample_categories(atom_logits, generator)
        bond_candidates = _sample_categories(bond_logits, generator)
        reveal_probability = schedule.conditional_reveal_probability(start, end)
        reveal_atoms = (
            (atom_ids == vocabulary.mask_id)
            & atom_valid
            & (
                torch.rand(atom_ids.shape, device=device, generator=generator)
                < reveal_probability[:, None]
            )
        )
        reveal_bonds = (
            (bond_ids == BondToken.MASK)
            & bond_valid
            & (
                torch.rand(bond_ids.shape, device=device, generator=generator)
                < reveal_probability[:, None]
            )
        )
        atom_ids[reveal_atoms] = atom_candidates[reveal_atoms]
        bond_ids[reveal_bonds] = bond_candidates[reveal_bonds]

    if torch.any(atom_ids[atom_valid] == vocabulary.mask_id):
        raise RuntimeError("Base sampling ended with masked atoms")
    if torch.any(bond_ids[bond_valid] == BondToken.MASK):
        raise RuntimeError("Base sampling ended with masked bonds")
    graphs = []
    for row, count in enumerate(n_atoms):
        edge_count = count * (count - 1) // 2
        graphs.append(
            TokenGraph(
                atom_ids=tuple(int(value) for value in atom_ids[row, :count].cpu()),
                bond_ids=tuple(int(value) for value in bond_ids[row, :edge_count].cpu()),
            )
        )
    diagnostics = BaseSampleDiagnostics(
        steps=steps,
        model_forwards=model_forwards,
        revealed_atoms=sum(n_atoms),
        revealed_bonds=sum(count * (count - 1) // 2 for count in n_atoms),
    )
    return graphs, diagnostics
