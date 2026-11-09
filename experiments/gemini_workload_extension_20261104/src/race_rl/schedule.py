"""Deterministic resource calendars and precedence-based slack diagnostics."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Job:
    id: str
    dependencies: tuple[str, ...]
    resources: tuple[str, ...]
    duration_s: float
    kind: str = "compute"
    energy: tuple[tuple[str,float], ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("invalid job duration")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("duplicate resource reservation")
        if any(not math.isfinite(e) or e < 0 for _,e in self.energy):
            raise ValueError("invalid energy")


@dataclass(frozen=True)
class Event:
    job: Job
    start_s: float
    finish_s: float


@dataclass(frozen=True)
class Schedule:
    events: tuple[Event, ...]

    @property
    def event_map(self): return {e.job.id:e for e in self.events}

    @property
    def makespan_s(self): return max((e.finish_s for e in self.events),default=0.0)

    def validate(self):
        by_id = self.event_map
        for e in self.events:
            if any(by_id[d].finish_s > e.start_s + 1e-18 for d in e.job.dependencies):
                raise ValueError("dependency violation")
        calendars = {}
        for e in self.events:
            if e.finish_s > e.start_s:
                for r in e.job.resources:
                    calendars.setdefault(r,[]).append(e)
        for r,spans in calendars.items():
            spans.sort(key=lambda e:(e.start_s,e.job.id))
            if any(a.finish_s > b.start_s + 1e-18 for a,b in zip(spans,spans[1:])):
                raise ValueError(f"resource conflict: {r}")
        return True


def earliest(calendars,resources,ready,duration):
    cursor = ready
    if duration == 0:
        return cursor
    while True:
        conflicts = [b for r in resources for a,b in calendars.get(r,())
                     if a < cursor+duration-1e-18 and b > cursor+1e-18]
        if not conflicts:
            return cursor
        cursor = max(conflicts)


def schedule(jobs):
    jobs = tuple(jobs)
    by_id = {j.id:j for j in jobs}
    if len(by_id) != len(jobs) or any(d not in by_id or d == j.id for j in jobs for d in j.dependencies):
        raise ValueError("duplicate/unknown/self-referential job ID")
    pending = dict(by_id)
    done,calendars = {},{}
    while pending:
        choices = []
        for j in pending.values():
            if all(d in done for d in j.dependencies):
                release = max((done[d].finish_s for d in j.dependencies),default=0.0)
                start = earliest(calendars,j.resources,release,j.duration_s)
                choices.append((start,j.id))
        if not choices:
            raise ValueError("cyclic job dependencies")
        start,name = min(choices)
        j = pending.pop(name)
        e = Event(j,start,start+j.duration_s)
        done[name] = e
        if j.duration_s:
            for r in j.resources:
                calendars.setdefault(r,[]).append((start,e.finish_s))
                calendars[r].sort()
    result = Schedule(tuple(done.values()))
    result.validate()
    return result


def criticality(result):
    """All critical jobs in data + actual resource ordering DAG; a proxy, not causality."""
    events = result.event_map
    order = {e.job.id:i for i,e in enumerate(result.events)}
    predecessors = {k:set(e.job.dependencies) for k,e in events.items()}
    resources = {}
    for e in result.events:
        if e.job.duration_s > 0:
            for r in e.job.resources:
                resources.setdefault(r,[]).append(e)
    for spans in resources.values():
        spans.sort(key=lambda e:(e.start_s,order[e.job.id]))
        for a,b in zip(spans,spans[1:]):
            predecessors[b.job.id].add(a.job.id)
    pending = set(events)
    earliest_start,sequence = {},[]
    while pending:
        ready = sorted(k for k in pending if predecessors[k] <= set(earliest_start))
        if not ready:
            raise ValueError("critical precedence cycle")
        for k in ready:
            earliest_start[k] = max((earliest_start[d]+events[d].job.duration_s for d in predecessors[k]),default=0)
            sequence.append(k)
            pending.remove(k)
    successors = {k:set() for k in events}
    for k,deps in predecessors.items():
        for d in deps: successors[d].add(k)
    latest = {}
    for k in reversed(sequence):
        latest[k] = min((latest[s] for s in successors[k]),default=result.makespan_s)-events[k].job.duration_s
    return {k:{"earliest_start_s":earliest_start[k],"latest_start_s":latest[k],
               "slack_s":max(0,latest[k]-earliest_start[k]),
               "critical":math.isclose(latest[k],earliest_start[k],abs_tol=1e-15,rel_tol=1e-9),
               "predecessors":sorted(predecessors[k])} for k in sequence}


def peak_bytes(intervals):
    changes = []
    for a,b,size in intervals:
        if b < a or size < 0 or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("invalid buffer lifetime")
        if b > a and size:
            changes.extend(((a,size),(b,-size)))
    live,peak = 0,0
    for _,delta in sorted(changes):
        live += delta
        peak = max(peak,live)
    return peak
