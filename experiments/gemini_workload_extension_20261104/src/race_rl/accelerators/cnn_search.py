"""Sequential seven-CNN hierarchy search; compact, resumable research records."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import html
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from ..serialization import save_json,code_hashes,digest,plain
from ..recording import CompactRecorder,packed,policy_hash,assessment_better
from ..requirements import PPARequirements
from ..frontend import SUPPORTED_MODELS,make_torchvision_model,from_torch
from ..training import RACETrainer
from ..stopping import ObjectivePlateau,PlateauSettings
from .memory_hierarchy import MemoryHierarchy
from .gemmini_model import PlatformScenario
from .gemmini_config import verify_reference
from .gemmini_ppa import GemminiCostProfile
from .cnn_ppa import CNNCostProfile,load_cnn_cost_profile,cnn_hardware_record
from .cnn_quantization import calibrate,execute_quantized
from .cnn_environment import GemminiCNNEnvironment
from .cnn_design import CNNDesign,LayerPlan

ROOT=Path(__file__).resolve().parents[3]

def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

def fixture(memory):
    fields=GemminiCostProfile.__dataclass_fields__
    coefficients={key:1e-12 for key in fields if key not in ('name','evidence_status','sources','memory_hierarchy_identity')}
    coefficients.update(clock_hz=100e6,chiplet_area_mm2=1,chiplet_static_power_w=.001,dram_static_power_w=.001)
    base=GemminiCostProfile(name='CNN_TEST_FIXTURE_NOT_REAL_PPA',evidence_status='TEST_FIXTURE',
         sources={key:'Synthetic integration constant; no fabrication/measurement claim' for key in coefficients},
         memory_hierarchy_identity=memory.identity,**coefficients)
    return CNNCostProfile(base,1e-12,'Synthetic vector test constant; not hardware data',cnn_hardware_record(memory)['identity'])


def design_view(design,record,compliant=False,memory=None):
    from ..mesh import mesh_shape
    count=design['compute_count'];nr,nc=mesh_shape(count)
    assignment={f'c{i}':{} for i in range(count)}
    for layer in (record or {}).get('operator_placement',()):
        for node,n in layer['tiles_by_node'].items():assignment[node][layer['operator']]=n
    nodes=[{'id':f'c{i}','grid_x':i%nc,'grid_y':i//nc,'x_mm':i%nc,'y_mm':i//nc,
            'width_mm':None,'height_mm':None,'tile_count':sum(assignment[f'c{i}'].values()),
            'operators':assignment[f'c{i}'],'tiles':[]} for i in range(count)]
    directed={(x['from'],x['to']):x['bytes'] for x in (record or {}).get('directed_links',())}
    flows={(x['from'],x['to']):x.get('flows',0) for x in (record or {}).get('directed_links',())}
    links=[]
    scenario=(record or {}).get('platform_scenario',asdict(PlatformScenario()))
    for i in range(count):
        for other in (i+1 if i%nc+1<nc else None,i+nc if i//nc+1<nr else None):
            if other is None:continue
            links.append({'id':f'c{i}-c{other}','a':f'c{i}','b':f'c{other}',
                          'bytes_per_cycle':scenario['mesh_bytes_per_cycle'],'bandwidth_Bps':None,'flows':flows.get((i,other),0)+flows.get((other,i),0),
                          'a_to_b_bytes':directed.get((i,other),0),'b_to_a_bytes':directed.get((other,i),0)})
    traffic=(record or {}).get('traffic',{})
    operands=traffic.get('by_operand',{})
    return {'backend':'gemmini_cnn','coordinate_unit':'grid','physical_feasible':record is not None,
            'compliant':compliant,'compute_count':count,'nodes':nodes,'links':links,
            'memory_ports':[{'id':'dram0','attached_to':'c0','read_bytes':sum(v for k,v in operands.items() if k!='output'),
                             'write_bytes':operands.get('output',0)}],
            'operator_placement':(record or {}).get('operator_placement',()),
            'largest_transfers':(record or {}).get('largest_transfers',()),'transfer_count':(record or {}).get('transfer_count',0),
            'total_mesh_hop_bytes':traffic.get('directed_link_byte_hops',0),'hierarchy_access_bytes':traffic.get('storage_access_bytes',{}),
            'modeled_cycles':(record or {}).get('modeled_cycles'),'memory_hierarchy':(memory or MemoryHierarchy()).record(),
            'traffic_definition':'cumulative modeled directed-hop payload; coordinates are mesh grid indices, not measured die geometry'}


def prepare_model(name,size,seed,folder):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    example=torch.randn(1,3,size,size,generator=torch.Generator().manual_seed(seed+1),dtype=torch.float32)
    imported=from_torch(make_torchvision_model(name,seed),example,name)
    bundle=calibrate(imported,example)
    # Release duplicate float model weights before integer numeric execution.
    del imported;gc.collect()
    _,verification=execute_quantized(bundle,example.numpy())
    save_json(folder/'quantization.json',bundle.manifest)
    save_json(folder/'numerical_validation.json',verification)
    workload=bundle.workload
    save_json(folder/'workload.json',workload)
    del bundle,example;gc.collect()
    return workload,verification


def worker(run,name):
    protocol=read(run/'protocol.json');config=protocol['config']
    if protocol['source_hashes']!=code_hashes():raise ValueError('source changed; cannot resume this experiment')
    folder=run/name;folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();save_json(folder/'status.json',{'status':'RUNNING','phase':'quantization'})
    workload,numerical=prepare_model(name,config['input_size'],config['seed'],folder)
    memory=MemoryHierarchy.from_record(config['memory']);scenario=PlatformScenario.from_record(config['platform'])
    mode=config['mode']
    profile=fixture(memory) if mode=='integration' else load_cnn_cost_profile(config['cost_profile'],memory=memory) if config['cost_profile'] else None
    requirements=PPARequirements(**config['requirements'])
    env=GemminiCNNEnvironment(workload,requirements=requirements,memory=memory,platform=scenario,
         cost_profile=profile,allow_test_fixture=mode=='integration',budget=config['interactions'],max_scope=config['max_scope'])
    env.reset() # Structural state/scorer only; zero model or PPA evaluations.
    identity=digest({'protocol':protocol['identity'],'model':name,'environment':env.experiment_identity})
    recorder=CompactRecorder(folder/'history.sqlite',identity)
    interrupted=False
    try:
        checkpoint=recorder.checkpoint()
        trainer=RACETrainer.load(checkpoint,experiment_identity=identity) if checkpoint else RACETrainer(
            seed=config['seed'],width=32,precision='float64',policy_backend='gemmini_cnn')
        env.progress_path=folder/'progress.json'
        env.progress_context={'model':name,'phase':'train','completed_actions':0,'elapsed_s':0}
        winner=recorder.get('winner');all_steps=sum(row['oracle_requests'] for row in recorder.sessions())
        plateau=ObjectivePlateau(PlateauSettings(**config['early_stopping'])) if config.get('early_stopping') else None
        if plateau:
            for row in recorder.sessions():
                if row['phase']=='train':plateau.update(row.get('archive_assessment',row['assessment']),row['episode_number'])
        stop_reason=recorder.get('stop_reason','EPISODE_LIMIT')
        def snapshot():
            return {'design':asdict(env.best_design),'model':env.best_record.model_feedback,
                    'evaluation':asdict(env.best_record),'assessment':env.scorer.assess(env.best_record)} if env.best_record else None
        def store(phase,seed,episode,stats,learn):
            nonlocal winner,all_steps
            candidate=snapshot()
            if learn and candidate and assessment_better(candidate['assessment'],winner['assessment'] if winner else None):winner=candidate
            all_steps+=env.oracle.requests
            stats={**stats,'phase':phase,'wall_time_s':time.perf_counter()-round_started,
                   'assessment':env.scorer.assess(env.best_record),'policy_hash_after':policy_hash(trainer.policy),
                   'archive_assessment':winner['assessment'] if winner else env.scorer.assess(None),'phase_seed':seed}
            buffer=io.BytesIO()
            if learn:trainer.save(buffer,experiment_identity=identity)
            with recorder.db:
                recorder.db.execute('INSERT INTO episodes VALUES (?,?,?,?,?)',(phase,seed,episode,json.dumps(plain(stats),allow_nan=False),packed(env.trace)))
                recorder.put('winner',winner)
                if learn:recorder.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('checkpoint',buffer.getvalue()))
            save_json(folder/'training_curve.json',[row for row in recorder.sessions() if row['phase']=='train'])
            save_json(folder/'status.json',{'status':'RUNNING','phase':phase,'training_completed':trainer.episodes,
                      'completed_actions':all_steps,'elapsed_s':time.perf_counter()-started})
        if mode=='inspect':
            o=env.reset();before=policy_hash(trainer.policy)
            action,_,_=trainer.policy.select(o)
            _,r,_=env.step(action)
            stats={'oracle_requests':1,'oracle_evaluations':env.oracle.evaluations,'parameter_updates':0,
                   'policy_hash_before':before,'policy_hash_after':policy_hash(trainer.policy)}
            with recorder.db:
                recorder.db.execute('INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?)',('inspect',config['seed'],1,json.dumps(stats),packed(env.trace)))
            selected=asdict(action.design);feedback=env.current_feedback
            assessment=env.scorer.assess(env.current_record)
            if feedback is not None and profile is None:
                assessment={**assessment,'physical_feasible':True,
                            'violations':['模型容量／資料流檢查通過；PPA 成本資料缺值，未評分、未訓練。']}
            all_steps=1
            report_status='PPA_UNAVAILABLE' if not profile else 'FROZEN_MODEL_INSPECTION'
        else:
            from ..model_suite import frozen_episode
            for phase in ('baseline','train','final'):
                if phase=='train' and stop_reason=='OBJECTIVE_PLATEAU':continue
                seeds=config['evaluation_seeds'] if phase!='train' else [config['seed']]
                for seed in seeds:
                    rounds=range(1,config['episodes']+1) if phase=='train' else (1,)
                    for episode in rounds:
                        if recorder.has(phase,seed,episode):continue
                        env.progress_context={'model':name,'phase':phase,'completed_actions':all_steps,'elapsed_s':time.perf_counter()-started}
                        round_started=time.perf_counter();before=policy_hash(trainer.policy)
                        stats=trainer.run_episode(env) if phase=='train' else frozen_episode(trainer,env,seed)
                        stats['policy_hash_before']=before
                        store(phase,seed,episode,stats,phase=='train')
                        if phase=='train' and plateau and plateau.update(winner['assessment'] if winner else env.scorer.assess(None),episode):
                            stop_reason='OBJECTIVE_PLATEAU'
                            with recorder.db:recorder.put('stop_reason',stop_reason)
                            break
            selected=winner['design'] if winner else asdict(env.current_design)
            feedback=winner['model'] if winner else env.current_feedback
            assessment=winner['assessment'] if winner else env.scorer.assess(None)
            report_status='INTEGRATION_TEST_ONLY' if mode=='integration' else 'DECLARED_PROFILE_TRAINING_COMPLETED'
        # Best design evaluated during training, not selected from final tests.
        rows=recorder.sessions()
        result={'status':report_status,'model':name,'backend':'race_gemmini_cnn_hierarchy_v1',
            'source_identity':identity,'execution_platform':protocol['execution_platform'],
            'wall_time_s':time.perf_counter()-started,'single_agent':True,'algorithm':'reinforce','critic_used':False,
            'evaluations_before_first_action':0,'completed_actions':all_steps,
            'training_episodes_completed':trainer.episodes,'policy_updates':sum(row['parameter_updates'] for row in rows),
            'policy_hash':policy_hash(trainer.policy),'numerical_validation':numerical,
            'assessment':assessment,'selected_design':selected,'selected_model_feedback':feedback,'sessions':rows,
            'cost_evidence_status':profile.base.evidence_status if profile else 'UNAVAILABLE',
            'physically_calibrated':False,'accuracy_validated':False,'timing_status':'LAYER_BARRIER_BOTTLENECK_ESTIMATE_UNCALIBRATED',
            'interpretation':'full CNN software/analytical integration; not silicon PPA, accuracy, optimality, or convergence proof'}
        save_json(folder/'report.json',result)
        save_json(folder/'design_view.json',design_view(selected,feedback,assessment['feasible'],memory))
        vendor=env.scorer.report(env.best_record)
        vendor.update(assessment=assessment,status='FEASIBLE_DESIGN_FOUND' if assessment['feasible'] else 'PPA_UNAVAILABLE' if not profile else 'NO_FEASIBLE_DESIGN_FOUND',
                      performance_definition='analytical single-inference cycle estimate / supplied clock; not an upper-bound proof')
        if feedback is None and env.trace:vendor['assessment']['violations']=list(env.trace[-1]['violations']) or ['PPA 成本資料缺值，未評分。']
        save_json(folder/'vendor_report.json',vendor)
        save_json(folder/'summary.json',{'backend':'gemmini_cnn','selected_compute_count':selected['compute_count'],
            'episodes_completed':trainer.episodes,'search_stop_reason':'FROZEN_INSPECTION' if mode=='inspect' else stop_reason,
            'model_cycles':feedback['modeled_cycles'] if feedback else None,'completed_actions':all_steps,
            'ppa_available':profile is not None,'status':report_status})
        save_json(folder/'status.json',{'status':'COMPLETED','training_completed':trainer.episodes,'completed_actions':all_steps,
                                      'elapsed_s':time.perf_counter()-started})
    except KeyboardInterrupt:
        interrupted=True
        save_json(folder/'status.json',{'status':'INTERRUPTED','completed_actions':sum(row['oracle_requests'] for row in recorder.sessions()),
                                      'training_completed':trainer.episodes,'partial_episode_discarded':True})
        raise
    finally:recorder.close()


def batch_html(reports,protocol):
    rows=[]
    for report in reports:
        a=report['assessment'];metrics=a['metrics']
        values=(report['model'],report['status'],report['training_episodes_completed'],report['completed_actions'],
                f"{report['wall_time_s']:.2f}",report['selected_design']['compute_count'],
                *(metrics[key] for key in ('average_power_w','latency_s','area_mm2')))
        rows.append('<tr>'+''.join('<td>'+html.escape('缺值' if value is None else str(value))+'</td>' for value in values)+'</tr>')
    return '<!doctype html><meta charset="utf-8"><title>RACE-RL CNN 記憶體階層實驗</title><style>body{font-family:Times New Roman,DFKai-SB,serif;max-width:1200px;margin:40px auto;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bfcdd4;padding:8px;text-align:left}</style><h1>RACE-RL CNN 記憶體階層實驗</h1><p>單一 REINFORCE Agent；H／W／輸出通道切分、K blocking、GB 駐留與 mesh mapping。完整 action 後才評估；新向量單元、時序與 PPA 尚未經實體校準。</p><p>模式：'+html.escape(protocol['config']['mode'])+'。baseline／final 為等預算凍結策略測試；訓練互動另計。TEST_FIXTURE 結果只能驗證程式串接。量化誤差對照隨機權重 FP32 網路，沒有資料集準確率證据。</p><table><tr>'+''.join('<th>'+label+'</th>' for label in ('模型','狀態','訓練回合','全部互動','總秒數','顆數','平均功耗 W','延遲 s','面積 mm²'))+'</tr>'+''.join(rows)+'</table><p>各模型 report.json 記錄數值誤差、每回合成效、凍結測試、平台、時間與最佳方案。history.sqlite 保存壓縮的逐次實際回饋及唯一 checkpoint；重跑必須使用相同來源／設定身份。</p>'


def main(argv=None):
    if os.name=='nt':signal.signal(signal.SIGBREAK,signal.default_int_handler)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models','--model',nargs='+',default=['all'])
    parser.add_argument('--input-size',type=int,default=64)
    parser.add_argument('--episodes',type=int,default=60)
    parser.add_argument('--interactions',type=int,default=64)
    parser.add_argument('--max-scope',type=int,default=2)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--early-stop',action='store_true')
    parser.add_argument('--min-episodes',type=int,default=20)
    parser.add_argument('--patience',type=int,default=15)
    parser.add_argument('--min-improvement',type=float,default=.001)
    parser.add_argument('--evaluation-seeds',type=int,nargs='+',default=[17,29,43])
    parser.add_argument('--memory',type=Path,default=ROOT/'configs/memory_hierarchy.json')
    parser.add_argument('--platform',type=Path,default=ROOT/'configs/gemmini_platform.json')
    parser.add_argument('--ppa-profile',type=Path)
    parser.add_argument('--requirements',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect',action='store_true')
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--integration-smoke',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/runs/cnn_hierarchy')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker',choices=SUPPORTED_MODELS,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.resume:
        supplied=list(sys.argv[1:] if argv is None else argv)
        for option in supplied:
            if option.startswith('--') and option not in ('--resume','--out'):
                parser.error('resume uses the frozen protocol; only --out and --resume may be supplied')
    if args.worker:
        try:worker(args.out,args.worker)
        except Exception as error:
            save_json(args.out/args.worker/'status.json',{'status':'FAILED','error':str(error)})
            raise
        return 0
    try:
        models=list(SUPPORTED_MODELS) if args.models==['all'] else args.models
        if not models or len(models)!=len(set(models)) or any(name not in SUPPORTED_MODELS for name in models):raise ValueError('choose all or distinct supported CNN names')
        if not 32<=args.input_size<=512 or args.episodes<1 or args.interactions<1 or not 1<=args.max_scope<=4 or not 0<=args.seed<2**31:
            raise ValueError('invalid input size/episode/interaction/scope/seed settings')
        if not args.evaluation_seeds or len(set(args.evaluation_seeds))!=len(args.evaluation_seeds) or any(not 0<=s<2**31 for s in args.evaluation_seeds):raise ValueError('distinct evaluation seeds required')
        memory_record=read(args.memory);memory=MemoryHierarchy.from_record(memory_record)
        platform_record=read(args.platform);scenario=PlatformScenario.from_record(platform_record)
        cost_record=read(args.ppa_profile) if args.ppa_profile else None
        if args.ppa_profile and not cost_record:raise ValueError('invalid cost profile JSON')
        profile=load_cnn_cost_profile(cost_record,memory=memory) if cost_record else None
        requirements=PPARequirements(**read(args.requirements)) if args.requirements else PPARequirements()
        if args.integration_smoke:
            if profile or args.requirements:raise ValueError('integration mode uses only TEST_FIXTURE costs/limits')
            requirements=PPARequirements(1,1,16)
            episodes,interactions=2,4
            selected_mode='integration'
        else:
            selected_mode='train' if args.train else 'inspect'
            episodes,interactions=args.episodes,args.interactions if args.train else 1
        if args.train:
            if not profile:raise ValueError('CNN training requires sourced costs for all memory levels and the added vector unit; no legacy fallback')
            requirements.require_vendor_inputs()
        if profile and scenario.clock_hz not in (None,profile.base.clock_hz):raise ValueError('platform/cost clock mismatch')
        audit=verify_reference(ROOT)
        if audit['status']!='PASSED':raise ValueError('Gemmini reference source audit failed')
        if args.resume:
            protocol=read(args.out/'protocol.json')
            if not protocol or protocol['source_hashes']!=code_hashes():raise ValueError('no matching frozen protocol to resume')
        else:
            if args.out.exists() and any(args.out.iterdir()):raise ValueError('choose a fresh --out or explicitly --resume')
            config={'models':models,'input_size':args.input_size,'episodes':episodes,'interactions':interactions,
                    'candidate_compute_counts':list(range(1,17)),'initial_count':2,'initial_layer_plan':asdict(LayerPlan()),
                    'max_scope':args.max_scope,'seed':args.seed,'evaluation_seeds':args.evaluation_seeds,
                    'early_stopping':plain(PlateauSettings(args.min_episodes,args.patience,args.min_improvement)) if args.early_stop and selected_mode=='train' else None,
                    'memory':memory_record,'platform':platform_record,'cost_profile':cost_record,
                    'requirements':plain(requirements),'mode':selected_mode,'algorithm':'reinforce'}
            from ..experiment_diagnostics import platform_snapshot
            execution=platform_snapshot()
            execution['precision']={'policy':'float64','source_cnn':'float32','hardware_operands':'INT8',
                                    'hardware_accumulators':'INT32','vector_reference':'FP32'}
            protocol={'created_at':datetime.now(timezone.utc).isoformat(),'config':config,'source_hashes':code_hashes(),
                      'execution_platform':execution,'source_audit':audit,'hardware':cnn_hardware_record(memory),
                      'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            protocol['identity']=digest(protocol);save_json(args.out/'protocol.json',protocol)
    except (TypeError,ValueError,KeyError,OSError) as error:parser.error(str(error))
    reports=[];models=protocol['config']['models'];cfg=protocol['config']
    print(f"單一 REINFORCE；{len(models)} 個模型依序；模式 {cfg['mode']}；輸入 {cfg['input_size']}；scope={cfg['max_scope']}；RL 選 1–16 顆。",flush=True)
    if cfg['mode']=='inspect':print('成本缺值時只檢視一次完整動作；PPA 留空、不更新策略。',flush=True)
    if cfg['mode']=='integration':print('TEST_FIXTURE：只驗證完整 CNN／RL 串接，不能作為真實 PPA 或收斂證據。',flush=True)
    child=None
    try:
        for i,name in enumerate(models,1):
            folder=args.out/name;status=read(folder/'status.json',{})
            if status.get('status')!='COMPLETED':
                folder.mkdir(parents=True,exist_ok=True)
                with (folder/'execution.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','race_rl.accelerators.cnn_search','--worker',name,'--out',str(args.out.resolve())],
                        stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1'},
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
                    began=time.perf_counter()
                    while child.poll() is None:
                        if sys.stdout.isatty():
                            current=read(folder/'status.json',{});live=read(folder/'progress.json',{})
                            done=live.get('completed_actions',current.get('completed_actions',0))+live.get('requests',0)
                            total=1 if cfg['mode']=='inspect' else (cfg['episodes']+2*len(cfg['evaluation_seeds']))*cfg['interactions']
                            elapsed=time.perf_counter()-began;eta=elapsed/done*max(0,total-done) if done else None
                            line=f"[{i}/{len(models)}] {name:<12} {live.get('stage',current.get('phase','初始化')):<20} 互動 {done:>5}/{total:<5} 耗時 {elapsed:>7.0f}s ETA {f'{eta:.0f}s' if eta is not None else '估計中'}"
                            print('\r\x1b[2K'+line,end='',flush=True)
                        time.sleep(.5)
                    if sys.stdout.isatty():print()
                    if child.returncode:raise RuntimeError(f'{name} 未完成；執行紀錄：{folder/"execution.log"}')
                    child=None
            report=read(folder/'report.json')
            if not report:raise RuntimeError('completed report missing')
            reports.append(report)
            print(f"[{i}/{len(models)}] {name:<12} 完成 {report['training_episodes_completed']:>2} 回合；全部互動 {report['completed_actions']:>5}；總耗時 {report['wall_time_s']:.1f}s",flush=True)
            save_json(args.out/'batch_summary.json',{'status':'RUNNING','models':[{'model':r['model'],'status':r['status'],'wall_time_s':r['wall_time_s']} for r in reports]})
        save_json(args.out/'batch_summary.json',{'status':'COMPLETED','models':[{'model':r['model'],'status':r['status'],'wall_time_s':r['wall_time_s']} for r in reports],
                  'total_wall_time_s':sum(r['wall_time_s'] for r in reports),'source_identity':protocol['identity']})
        (args.out/'report.html').write_text(batch_html(reports,protocol),encoding='utf-8')
        print('批次完成；報告：'+str(args.out/'report.html'))
    except KeyboardInterrupt:
        if child and child.poll() is None:
            try:child.send_signal(signal.CTRL_BREAK_EVENT if os.name=='nt' else signal.SIGINT)
            except ProcessLookupError:pass
            except OSError:
                if os.name!='nt':raise
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,timeout=5)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.terminate();child.wait()
        print('\n已停止；SQLite 保留完整回合與 checkpoint。使用相同目錄加 --resume。')
        return 130
    return 0


if __name__=='__main__':raise SystemExit(main())
