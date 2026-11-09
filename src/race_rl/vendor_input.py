"""Validate vendor inputs before model import, then freeze a PPA task."""
import json
from .requirements import PPARequirements,PREFERENCE_LABELS


def prompt_requirements(read=None,write=None):
    read = read or input
    write = write or print
    write('請輸入 PPA 限制值：平均功耗 W、單次推論延遲 ms、運算晶粒面積 mm²。')
    def positive(label):
        while True:
            try:
                value = float(read(label))
                import math
                if math.isfinite(value) and value > 0: return value
            except ValueError: pass
            write('請輸入大於零的有限數值。')
    power = positive('功耗上限（W）：')
    latency = positive('推論延遲上限（ms）：')/1000
    area = positive('運算晶粒面積上限（mm²）：')
    names = tuple(PREFERENCE_LABELS)
    write('\n'.join(f'{i+1}. {PREFERENCE_LABELS[name]}' for i,name in enumerate(names)))
    while True:
        choice = read('偏好選項（1–5）：').strip()
        if choice in ('1','2','3','4','5'): break
        write('請選擇 1–5。')
    preference = names[int(choice)-1]
    weights = None
    if preference == 'custom':
        while True:
            try:
                weights = tuple(float(v) for v in read('依序輸入功耗、效能、面積權重（空白分隔）：').split())
                return PPARequirements(power,latency,area,preference,weights)
            except ValueError:
                write('請輸入三個非負有限權重，總和須大於零。')
    return PPARequirements(power,latency,area,preference)


def load_requirements(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload,dict) or set(payload)-{'max_power_w','max_latency_s','max_area_mm2','preference','weights'}:
        raise ValueError('requirements file must contain only PPA limits, preference and optional custom weights')
    if 'preference' not in payload: raise ValueError('vendor must choose one preference')
    return PPARequirements(**payload).require_vendor_inputs()


def requirements_from_args(args):
    supplied = (args.max_power_w,args.max_latency_s,args.max_area_mm2,args.preference,args.ppa_weights)
    if args.interactive:
        if args.requirements or any(v is not None for v in supplied):
            raise ValueError('interactive input cannot be mixed with a file or PPA flags')
        return prompt_requirements().require_vendor_inputs()
    if args.requirements:
        if any(v is not None for v in supplied):
            raise ValueError('choose a requirements file or PPA flags; do not mix them')
        return load_requirements(args.requirements)
    if args.preference is None: raise ValueError('choose a PPA preference or use --interactive / --requirements')
    return PPARequirements(args.max_power_w,args.max_latency_s,args.max_area_mm2,
        args.preference,tuple(args.ppa_weights) if args.ppa_weights is not None else None).require_vendor_inputs()
