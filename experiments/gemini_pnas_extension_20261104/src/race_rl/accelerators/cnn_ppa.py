"""Full CNN cost boundary includes the project-added vector unit and SRAM/RF."""
from dataclasses import dataclass,asdict
import math
from ..serialization import digest
from .gemmini_config import default_config
from .memory_hierarchy import MemoryHierarchy
from .gemmini_ppa import load_cost_profile


def cnn_hardware_record(memory=None):
    memory=memory or MemoryHierarchy()
    parameters={'memory_hierarchy':memory.record(),'reference':default_config().provenance(),
        'vector_unit':{'lanes':16,'precision':'FP32','operations_per_cycle':16,'status':'RESEARCH_EXTENSION'},
        'gb_allocator':{'activation_cache_bank':0,'gather_a_output_bank':1,'weight_banks':[2,3]},
        'pe_precision':'INT8 operands / INT32 accumulated output; 20-bit intermediate RTL not reproduced'}
    return {'identity':digest(parameters),'parameters':parameters,'ppa_calibrated':False,'rtl_implemented':False}


@dataclass(frozen=True)
class CNNCostProfile:
    base: object
    vector_op_j: float
    vector_op_source: str
    cnn_hardware_identity: str

    def validate(self,*,allow_test_fixture=False,memory=None):
        self.base.validate(allow_test_fixture=allow_test_fixture)
        if (isinstance(self.vector_op_j,bool) or not isinstance(self.vector_op_j,(int,float)) or
                not math.isfinite(self.vector_op_j) or self.vector_op_j<=0 or
                not isinstance(self.vector_op_source,str) or not self.vector_op_source.strip()):
            raise ValueError('CNN vector-unit energy coefficient and source required')
        if self.cnn_hardware_identity!=cnn_hardware_record(memory)['identity']:
            raise ValueError('CNN PPA profile does not match fixed vector/memory hardware')
        if self.base.memory_hierarchy_identity!=(memory or MemoryHierarchy()).identity:
            raise ValueError('CNN cost-profile memory identity mismatch')
        return self

    @property
    def identity(self):return digest(asdict(self))

    def assess(self,record):
        if record['model']!='race_gemmini_cnn_hierarchy_v1':raise ValueError('CNN cost requires CNN model feedback')
        result=self.base.assess(record)
        result['energy_components']['dma']=record['traffic']['dma_bus_bytes']*self.base.dma_byte_j
        result['energy_components']['vector']=record['operations']['vector_ops']*self.vector_op_j
        result['energy_j']=math.fsum(result['energy_components'].values())
        result['average_power_w']=result['energy_j']/result['latency_s']
        result.update(cost_profile=asdict(self),cost_profile_identity=self.identity,
                      timing_calibration_status=record['timing_status'],
                      area_boundary='all compute chiplets, GB/LB/RF/ACC, PE, added FP32 vector unit, DMA/router/PHY; excludes DRAM/package')
        return result


def load_cnn_cost_profile(record,*,allow_test_fixture=False,memory=None):
    if set(record)!= {'base','vector_op_j','vector_op_source','cnn_hardware_identity'}:
        raise ValueError('complete CNN cost profile required; GEMM-only or legacy coefficients are incompatible')
    return CNNCostProfile(load_cost_profile(record['base'],allow_test_fixture=allow_test_fixture),
                          record['vector_op_j'],record['vector_op_source'],record['cnn_hardware_identity']).validate(
                              allow_test_fixture=allow_test_fixture,memory=memory)
