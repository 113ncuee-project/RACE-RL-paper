"""Immutable tile placement and spill-to-memory residency mapping."""
from dataclasses import dataclass
from .tiles import Tile, validate_coverage


@dataclass(frozen=True)
class Mapping:
    tiles: tuple[Tile, ...]
    placement: tuple[tuple[str, str], ...]
    storage: tuple[tuple[str, str], ...]

    @property
    def placements(self): return dict(self.placement)

    @property
    def storages(self): return dict(self.storage)

    def validate(self, workload, hardware):
        workload.validate()
        hardware.validate()
        validate_coverage(workload,self.tiles)
        if len(self.placements) != len(self.placement) or set(self.placements) != {t.id for t in self.tiles}:
            raise ValueError("placement must cover each tile once")
        if not set(self.placements.values()) <= set(hardware.compute_ids):
            raise ValueError("illegal compute placement")
        required = set(workload.tensor_map) | set(workload.parameter_map)
        if len(self.storages) != len(self.storage) or set(self.storages) != required:
            raise ValueError("storage must cover tensors and weights exactly")
        if not set(self.storages.values()) <= set(hardware.memory_ids):
            raise ValueError("invalid memory endpoint")
        return True
