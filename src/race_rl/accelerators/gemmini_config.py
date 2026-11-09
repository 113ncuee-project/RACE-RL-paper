"""Pinned Gemmini defaultConfig facts, separate from research timing assumptions."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class GemminiConfig:
    tile_rows: int
    tile_columns: int
    mesh_rows: int
    mesh_columns: int
    input_bits: int
    weight_bits: int
    accumulator_bits: int
    array_output_bits: int
    scratchpad_bytes: int
    accumulator_bytes: int
    scratchpad_banks: int
    accumulator_banks: int
    scratchpad_singleported: bool
    accumulator_singleported: bool
    dma_bus_bits: int
    dma_max_bytes: int
    max_in_flight_requests: int
    load_queue_length: int
    store_queue_length: int
    execute_queue_length: int
    dataflows: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "dataflows", tuple(self.dataflows))
        for name, value in asdict(self).items():
            if name in ("dataflows", "scratchpad_singleported", "accumulator_singleported"):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"positive integer Gemmini parameter required: {name}")
        if self.rows != 16 or self.columns != 16 or (self.input_bits, self.weight_bits, self.accumulator_bits) != (8, 8, 32):
            raise ValueError("foundation executor supports only the pinned 16x16 INT8/INT32 profile")
        if self.scratchpad_banks != 4 or self.accumulator_banks != 2:
            raise ValueError("foundation bank allocator requires four scratchpad and two accumulator banks")
        if not self.scratchpad_singleported or self.accumulator_singleported:
            raise ValueError("foundation port model requires single-port SP and dual-port accumulator")
        if any(size % banks for size, banks in ((self.scratchpad_bytes, self.scratchpad_banks),
                                                (self.accumulator_bytes, self.accumulator_banks))):
            raise ValueError("bank capacities must divide evenly")
        if self.dma_bus_bits % 8 or self.dma_max_bytes < self.dma_beat_bytes:
            raise ValueError("invalid DMA width or maximum transaction size")

    @property
    def rows(self): return self.tile_rows * self.mesh_rows

    @property
    def columns(self): return self.tile_columns * self.mesh_columns

    @property
    def scratchpad_row_bytes(self): return self.columns * self.input_bits // 8

    @property
    def accumulator_row_bytes(self): return self.columns * self.accumulator_bits // 8

    @property
    def scratchpad_bank_bytes(self): return self.scratchpad_bytes // self.scratchpad_banks

    @property
    def accumulator_bank_bytes(self): return self.accumulator_bytes // self.accumulator_banks

    @property
    def dma_beat_bytes(self): return self.dma_bus_bits // 8

    def provenance(self):
        record = profile_record()
        return {"repository": record["repository"], "commit": record["commit"],
            "configuration": record["configuration"], "source_file": record["source_file"],
            "parameters_match_public_default": asdict(self) == asdict(default_config()),
            "profile_sha256": hashlib.sha256(profile_path().read_bytes()).hexdigest(),
            "rtl_timing_validated": False, "silicon_validated": False}


def profile_path(): return Path(__file__).with_name("gemmini_default.json")


def profile_record(): return json.loads(profile_path().read_text(encoding="utf-8"))


def default_config(): return GemminiConfig(**profile_record()["parameters"])


def verify_reference(root):
    """Verify both exact upstream bytes and the extracted defaultConfig facts offline."""
    root = Path(root)
    lock = json.loads((root / "gemmini.lock.json").read_text(encoding="utf-8"))
    profile = profile_record()
    failures = []
    if lock["commit"] != profile["commit"] or lock["repository"] != profile["repository"]:
        failures.append("profile source identity")
    for name, record in lock["reference_files"].items():
        path = root / "third_party/gemmini-reference" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(name)
    source = root / "third_party/gemmini-reference" / profile["source_file"]
    if source.is_file():
        # Parse only the defaultConfig block, never the later chip/dummy profiles.
        text = source.read_text(encoding="utf-8").split("val defaultConfig =", 1)[1].split("val dummyConfig", 1)[0]
        names = {"tile_rows": "tileRows", "tile_columns": "tileColumns", "mesh_rows": "meshRows",
            "mesh_columns": "meshColumns", "scratchpad_banks": "sp_banks", "accumulator_banks": "acc_banks",
            "dma_bus_bits": "dma_buswidth", "dma_max_bytes": "dma_maxbytes",
            "max_in_flight_requests": "max_in_flight_mem_reqs", "load_queue_length": "ld_queue_length",
            "store_queue_length": "st_queue_length", "execute_queue_length": "ex_queue_length"}
        patterns = {key: rf"\b{scala}\s*=\s*(\d+)" for key, scala in names.items()}
        for key, scala in (("input_bits", "inputType"), ("weight_bits", "weightType"),
                           ("accumulator_bits", "accType"), ("array_output_bits", "spatialArrayOutputType")):
            patterns[key] = rf"\b{scala}\s*=\s*SInt\((\d+)\.W\)"
        for key, scala in (("scratchpad_bytes", "sp_capacity"), ("accumulator_bytes", "acc_capacity")):
            patterns[key] = rf"\b{scala}\s*=\s*CapacityInKilobytes\((\d+)\)"
        for key, scala in (("scratchpad_singleported", "sp_singleported"),
                           ("accumulator_singleported", "acc_singleported")):
            patterns[key] = rf"\b{scala}\s*=\s*(true|false)"
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            expected = profile["parameters"][key]
            actual = (match.group(1) == "true" if key.endswith("singleported") else int(match.group(1))) if match else None
            if key.endswith("_bytes") and key in ("scratchpad_bytes", "accumulator_bytes") and actual is not None:
                actual *= 1024
            if actual != expected:
                failures.append("defaultConfig parameter: " + key)
        if not re.search(r"\bdataflow\s*=\s*Dataflow\.BOTH", text) or profile["parameters"]["dataflows"] != ["weight_stationary", "output_stationary"]:
            failures.append("defaultConfig dataflow")
    return {"dependency": "Gemmini reference", "commit": lock["commit"],
        "files_checked": len(lock["reference_files"]), "parameters_checked": 21,
        "mismatches": failures, "status": "PASSED" if not failures else "FAILED",
        "claim_scope": "source identity and configuration facts; not RTL/PPA validation"}
