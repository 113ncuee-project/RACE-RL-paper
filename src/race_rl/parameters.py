"""Parameter records carry units, origins and explicit lack of calibration."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Parameter:
    value: float
    unit: str
    source: str
    assumption: str
    sensitivity_range: tuple[float,float]
    calibration_status: str = "UNCALIBRATED"
    category: str = "EXPLORATORY_ASSUMPTION"

    def validate(self):
        a,b = self.sensitivity_range
        if not math.isfinite(self.value) or self.value < 0 or not 0 <= a <= self.value <= b or not math.isfinite(b):
            raise ValueError("invalid physical parameter range")
        if not self.unit or not self.source or not self.assumption:
            raise ValueError("parameter provenance required")
        return True


@dataclass(frozen=True)
class Profile:
    name: str
    operation_j: Parameter
    local_byte_j: Parameter
    external_byte_j: Parameter
    link_bit_j: Parameter

    def validate(self):
        for p,unit in [(self.operation_j,"J/op"),(self.local_byte_j,"J/B"),
                       (self.external_byte_j,"J/B"),(self.link_bit_j,"J/bit")]:
            p.validate()
            if p.unit != unit:
                raise ValueError("energy coefficient unit mismatch")
        return True


def exploratory_profile():
    def p(value,unit,meaning):
        return Parameter(value,unit,"configs/toy.json; deliberately synthetic test fixture",meaning,
                         (value/2,value*2))
    return Profile("toy_uncalibrated",p(1e-12,"J/op","one modeled MAC/vector/pool operation"),
        p(1e-12,"J/B","one local buffer byte access"),p(10e-12,"J/B","one external memory byte access"),
        p(0.5e-12,"J/bit","one directed link bit traversal"))
