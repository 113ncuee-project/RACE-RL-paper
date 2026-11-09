"""RACE-RL memory extension, explicitly separate from Gemmini defaultConfig.

One 16x16 PE cluster per chiplet. SRAM slots hold whole selected panels;
RF holds one 16x16 microtile. Accumulator holds the complete output block.
These capacities and port rates are research assumptions, not pinned RTL facts.
"""
from dataclasses import dataclass, asdict
from ..serialization import digest


@dataclass(frozen=True)
class MemoryHierarchy:
    lb_operand_slot_bytes: int = 4096
    lb_output_bytes: int = 4096
    lb_bank_bytes_per_cycle: int = 16
    gb_bank_bytes_per_cycle: int = 16
    rf_a_bytes: int = 256
    rf_b_bytes: int = 256
    rf_psum_bytes: int = 1024
    rf_operand_bytes_per_cycle: int = 256
    rf_psum_bytes_per_cycle: int = 1024
    accumulator_bank_bytes_per_cycle: int = 64

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(key+' must be a positive integer')

    @classmethod
    def from_record(cls, record):
        parameters = record['parameters']
        if set(parameters) != set(cls.__dataclass_fields__):
            raise ValueError('memory profile must explicitly supply every capacity and port rate')
        return cls(**parameters)

    @property
    def identity(self):
        return digest({'profile':'race_memory_hierarchy_v2', 'parameters':asdict(self)})

    def record(self):
        return {'profile':'race_memory_hierarchy_v2', 'identity':self.identity,
                'parameters':asdict(self), 'pe_clusters_per_chiplet':1,
                'lb_total_bytes':4*self.lb_operand_slot_bytes+self.lb_output_bytes,
                'rf_total_bytes':self.rf_a_bytes+self.rf_b_bytes+self.rf_psum_bytes,
                'status':'RESEARCH_EXTENSION_NOT_GEMMINI_DEFAULT_RTL',
                'capacity_provenance':'GB/ACC capacity from pinned Gemmini; LB/RF organization and all port rates are project assumptions',
                'ports':'GB and operand LB bank single port; output LB single shared port; RF and ACC read/write separate; copies reserve both endpoints'}
