"""Explicit cost-profile boundary; public Gemmini configuration is not PPA data."""
from dataclasses import dataclass, asdict
import math
from ..serialization import digest


@dataclass(frozen=True)
class GemminiCostProfile:
    name: str
    evidence_status: str
    sources: dict
    clock_hz: float
    chiplet_area_mm2: float
    memory_hierarchy_identity: str
    mac_j: float
    gb_byte_j: float
    lb_byte_j: float
    rf_byte_j: float
    accumulator_byte_j: float
    external_byte_j: float
    link_byte_hop_j: float
    dma_byte_j: float
    chiplet_static_power_w: float
    dram_static_power_w: float

    def validate(self, *, allow_test_fixture=False):
        if self.evidence_status not in ('DECLARED_ESTIMATE', 'RTL_DERIVED', 'MEASURED', 'TEST_FIXTURE'):
            raise ValueError('PPA profile requires a declared evidence status')
        if self.evidence_status == 'TEST_FIXTURE' and not allow_test_fixture:
            raise ValueError('test-only PPA coefficients require explicit allow_test_fixture')
        if not self.name or not isinstance(self.sources, dict): raise ValueError('PPA source metadata required')
        if not isinstance(self.memory_hierarchy_identity,str) or len(self.memory_hierarchy_identity)!=64:
            raise ValueError('PPA profile must identify the exact memory hierarchy')
        for key in self.__dataclass_fields__:
            if key in ('name', 'evidence_status', 'sources', 'memory_hierarchy_identity'): continue
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('missing or invalid PPA coefficient: '+key)
            if value == 0 and key not in ('chiplet_static_power_w', 'dram_static_power_w'):
                raise ValueError('positive PPA coefficient required: '+key)
            source = self.sources.get(key)
            if not isinstance(source, str) or not source.strip():
                raise ValueError('source or explicit assumption required for '+key)
        return self

    @property
    def identity(self): return digest(asdict(self))

    def assess(self, model_record):
        cycles, chiplets = model_record['modeled_cycles'], model_record['chiplets']
        latency = cycles/self.clock_hz
        operations, traffic = model_record['operations'], model_record['traffic']
        if model_record['memory_hierarchy']['identity'] != self.memory_hierarchy_identity:
            raise ValueError('PPA profile is for a different GB/LB/RF memory hierarchy')
        accesses = traffic['storage_access_bytes']
        # Each SRAM/RF endpoint read or write is counted once in the model
        # ledger. MAC cost excludes all register and memory accesses.
        components = {
            'mac': operations['padded_macs']*self.mac_j,
            'global_buffer': accesses['gb']*self.gb_byte_j,
            'local_buffer': accesses['lb']*self.lb_byte_j,
            'register_file': accesses['rf']*self.rf_byte_j,
            'accumulator': accesses['accumulator']*self.accumulator_byte_j,
            'external_memory': traffic['external_bytes']*self.external_byte_j,
            'interconnect': traffic['directed_link_byte_hops']*self.link_byte_hop_j,
            'dma': traffic['external_bytes']*self.dma_byte_j,
            'static': (chiplets*self.chiplet_static_power_w+self.dram_static_power_w)*latency,
        }
        energy = math.fsum(components.values())
        return {'energy_j': energy, 'latency_s': latency, 'average_power_w': energy/latency,
            'area_mm2': chiplets*self.chiplet_area_mm2, 'energy_components': components,
            'cost_profile': asdict(self), 'cost_profile_identity': self.identity,
            'coefficient_evidence_status': self.evidence_status,
            'confidence': 'TEST_ONLY' if self.evidence_status=='TEST_FIXTURE' else 'DECLARED_PROFILE_WITH_UNVALIDATED_TIMING',
            'timing_calibration_status': 'ANALYTICAL_UNVALIDATED',
            'area_boundary': 'compute chiplets including array, GB, LB, RF, accumulator, DMA/router/PHY; package and external DRAM excluded',
            'mac_coefficient_boundary': 'arithmetic datapath only; excludes all separately counted GB/LB/RF/ACC/DMA/network accesses'}


def load_cost_profile(record, *, allow_test_fixture=False):
    return GemminiCostProfile(**record).validate(allow_test_fixture=allow_test_fixture)
