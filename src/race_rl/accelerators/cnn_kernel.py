"""Closed-form repeated WS microprogram counts, checked against detailed jobs.

No candidate evaluation. This is the same fixed microprogram aggregated across
equal blocks, not a mapper or a replacement hardware choice.
"""
from .gemmini_model import padded,ceildiv


def kernel_counts(m,n,k,k_tile,memory,config):
    mp,np_,kp=padded(m),padded(n),padded(k)
    # One A panel per LB slot; two M16 microtiles share a B RF preload.
    mt=32
    if mt*min(padded(k),k_tile)>memory.lb_operand_slot_bytes:
        raise ValueError('CNN A panel exceeds LB capacity')
    if min(padded(k),k_tile)*np_>memory.lb_operand_slot_bytes:
        raise ValueError('CNN B panel exceeds LB capacity')
    if min(mt,mp)*np_*4>config.accumulator_bytes:
        raise ValueError('CNN partial sums exceed accumulator capacity')
    if (memory.rf_a_bytes<256 or memory.rf_b_bytes<256 or memory.rf_psum_bytes<1024
            or memory.lb_output_bytes<1024):
        raise ValueError('CNN microtile exceeds RF/output LB capacity')
    mg=ceildiv(m,mt);nb=np_//16;kb=kp//16
    a=mp*kp;b=kp*np_;c=mp*np_*4
    ar=a*nb;br=b*mg
    aw=kb*c;ac=(kb-1)*c
    mac=mp*np_*kp
    rf_compute=2*mac+4*(mac-mp*np_)+4*mac
    access={'gb':2*a+b*mg+2*c, # packed A fill/read, B read, INT32 output write/read
            'lb':a+b*mg+ar+br+2*(aw+ac+c),
            'rf':ar+br+aw+ac+rf_compute,'accumulator':aw+ac+c}
    moves={'gb->lb':a+b*mg,'lb->rf':ar+br+ac,'rf->lb':aw,
           'lb->accumulator':aw,'accumulator->lb':ac+c,'lb->gb':c}
    bw=memory.lb_bank_bytes_per_cycle
    copies=ceildiv(ar,min(bw,memory.rf_operand_bytes_per_cycle))+ceildiv(br,min(bw,memory.rf_operand_bytes_per_cycle))
    ps=2*ceildiv(ac,min(bw,memory.rf_psum_bytes_per_cycle,2*memory.accumulator_bank_bytes_per_cycle))+2*ceildiv(aw,min(bw,memory.rf_psum_bytes_per_cycle,2*memory.accumulator_bank_bytes_per_cycle))
    array=(mp//16)*nb*kb*46
    rf_cycles=max(ceildiv(mac,memory.rf_operand_bytes_per_cycle),ceildiv(4*(mac-mp*np_),memory.rf_psum_bytes_per_cycle),ceildiv(4*mac,memory.rf_psum_bytes_per_cycle))
    gb_copies=ceildiv(a,min(bw,memory.gb_bank_bytes_per_cycle))+ceildiv(b*mg,min(bw,2*memory.gb_bank_bytes_per_cycle))
    drain=ceildiv(c,min(bw,2*memory.accumulator_bank_bytes_per_cycle))+ceildiv(c,min(bw,2*memory.gb_bank_bytes_per_cycle))
    # Explicit research assumption: issue one A/B pair of panel-copy commands
    # at two cycles per K-panel. Larger panels amortize these commands.
    dispatch=2*mg*ceildiv(k,k_tile)
    service=mg*nb*kb*16+max(array,rf_cycles)+copies+ps+gb_copies+drain+dispatch
    return {'logical_macs':m*n*k,'padded_macs':mac,'a_packed_bytes':a,'b_panel_bytes':b*mg,
            'b_resident_bytes':b,'c_int32_bytes':c,'access':access,'moves':moves,
            'cycles':service,'panel_dispatch_cycles':dispatch,'working_set':{'gb_a_bytes':min(mt,mp)*min(kp,k_tile)+64,
            'lb_a_bytes':min(mt,mp)*min(kp,k_tile),'lb_b_bytes':min(kp,k_tile)*np_,
            'accumulator_bytes':min(mt,mp)*np_*4}}
