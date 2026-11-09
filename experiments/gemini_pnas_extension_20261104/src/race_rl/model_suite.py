"""Sequential, resumable CNN training with compact feedback and honest ETA."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil
import unicodedata

from .serialization import code_hashes, digest, plain, save_json
from .recording import CompactRecorder, policy_hash

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (2, 1, *range(3, 17))


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError): return default


def session_stats(env, stats, elapsed, before, after):
    return {**stats, 'wall_s': elapsed, 'assessment': env.scorer.assess(env.best_record),
            'selected_compute_count': len(env.best_hardware.compute_ids) if env.best_hardware else None,
            'action_counts': dict(Counter(a['action'].kind for a in env.trace)),
            'multi_tile_coupled_steps': sum(len(getattr(a['action'], 'scope', ())) > 1 for a in env.trace),
            'policy_hash_before': before, 'policy_hash_after': after, 'policy_changed': before != after}


def frozen_episode(trainer, env, seed):
    import torch
    import numpy as np
    import random
    state = torch.get_rng_state()
    numpy_state, python_state=np.random.get_state(),random.getstate()
    before = policy_hash(trainer.policy)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        stats = trainer.run_episode(env, learn=False)
    finally:
        torch.set_rng_state(state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    if policy_hash(trainer.policy) != before or stats['parameter_updates'] != 0:
        raise RuntimeError('frozen evaluation updated the policy')
    return stats


def suite_stages(config):
    seed=config['seed']
    stages=[('baseline',s,1) for s in config['evaluation_seeds']]
    for n in range(1,config['episodes']+1):
        stages.append(('train',seed,n))
        interval=config.get('evaluation_interval',0)
        if interval and n<config['episodes'] and n%interval==0:
            stages += [('periodic',s,n) for s in config['evaluation_seeds']]
    stages += [('final',s,1) for s in config['evaluation_seeds']]
    for method in config.get('comparisons',[]):
        stages += [(method,seed,n) for n in range(1,config['episodes']+1)]
    return stages


def worker(run, model):
    """One subprocess per model releases CNN and environment memory on exit."""
    import torch
    from .architecture import DesignEnvironment
    from .frontend import make_torchvision_model, from_torch
    from .model_validation import validate_model
    from .requirements import PPARequirements
    from .training import RACETrainer
    from .suite_report import export_model
    protocol = read_json(run/'protocol.json')
    config = protocol['config']
    if protocol['source_hashes'] != code_hashes(): raise ValueError('source changed; resume is refused')
    folder = run/model
    folder.mkdir(parents=True, exist_ok=True)
    status_path, progress_path = folder/'status.json', folder/'progress.json'
    seed = config['seed']
    started = time.perf_counter()
    old_status = read_json(status_path, {})
    status = {'model': model, 'status': 'RUNNING', 'phase': 'validation',
              'completed_actions': old_status.get('completed_actions',0),
              'training_completed': old_status.get('training_completed',0), 'phase_token': 'validation'}
    save_json(status_path, status)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    example = torch.randn(1, 3, config['input_size'], config['input_size'],
                          generator=torch.Generator().manual_seed(seed+1), dtype=torch.float32).double()
    imported = from_torch(make_torchvision_model(model, seed).double(), example, model)
    validation = validate_model(imported, example)
    save_json(folder/'numerical_validation.json', validation)
    if validation['status'] != 'PASSED': raise ValueError('numerical model validation failed')
    workload = imported.workload
    save_json(folder/'workload.json', workload)
    save_json(folder/'parameter_manifest.json', imported.parameter_manifest)
    # Environment uses the workload descriptor, not the large CNN tensors.
    del imported, example
    import gc
    gc.collect()
    identity = digest({'protocol': protocol['identity'], 'model': model, 'workload': workload})
    recorder = CompactRecorder(folder/'history.sqlite', identity)
    try:
        checkpoint = recorder.checkpoint()
        trainer = RACETrainer.load(checkpoint, experiment_identity=identity) if checkpoint else RACETrainer(
            seed=seed, width=32, precision='float64',lr=config.get('learning_rate',3e-4),
            entropy_coefficient=config.get('entropy_coefficient',0.0))
        from .feasibility import constraint_check,verify_task
        from .experiment_diagnostics import comparison_episode,platform_snapshot
        task=config.get('tasks',{}).get(model)
        requirements = PPARequirements(**(task['requirements'] if task else config['requirements']))
        feasibility=verify_task(workload,task,COUNTS) if task else constraint_check(workload,requirements,COUNTS)
        save_json(folder/'constraint_check.json',feasibility)
        if feasibility['status']=='PROVEN_IMPOSSIBLE':
            raise ValueError(' '.join(feasibility['warnings']))
        if not (folder/'platform_at_start.json').exists():
            save_json(folder/'platform_at_start.json',{'captured_at_utc':timestamp(),**platform_snapshot()})
        common = dict(compute_counts=COUNTS, budget=config['steps_per_episode'], max_scope=2,
                      requirements=requirements,reward_variant=config.get('reward_variant','dense'))
        previous = recorder.sessions()
        committed = sum(r['steps'] for r in previous)
        status.update(completed_actions=committed, training_completed=trainer.episodes)
        stages=suite_stages(config)
        for phase, sample_seed, number in stages:
            if recorder.has(phase, sample_seed, number): continue
            if phase == 'baseline' and trainer.episodes:
                raise RuntimeError('baseline is missing from a resumed trained policy')
            token = f'{phase}:{sample_seed}:{number}'
            status.update(phase=phase, phase_token=token, current_episode=number,
                          completed_actions=committed, training_completed=trainer.episodes)
            save_json(status_path, status)
            env = DesignEnvironment(workload, **common)
            env.progress_path = progress_path
            env.progress_context = {'phase': phase, 'phase_token': token, 'model': model}
            before = policy_hash(trainer.policy)
            episode_start = time.perf_counter()
            if phase=='train': stats=trainer.run_episode(env)
            elif phase in ('random','sa'):
                stats=comparison_episode(env,phase,sample_seed+number-1)
            else: stats=frozen_episode(trainer,env,sample_seed)
            stats = session_stats(env, stats, time.perf_counter()-episode_start,
                                  before, policy_hash(trainer.policy))
            stats['training_episode_at_measurement']=trainer.episodes
            if stats['steps'] != config['steps_per_episode']:
                raise RuntimeError('episode ended before its fixed interaction budget')
            recorder.record(phase, sample_seed, number, env, stats, trainer=trainer,
                            retain_winner=phase in ('train','random','sa'))
            committed += stats['steps']
            status.update(completed_actions=committed, training_completed=trainer.episodes,
                          phase_token='committed', elapsed_s=time.perf_counter()-started)
            save_json(status_path, status)
            a = stats['assessment']
            print(f'{model} {phase} {number}: steps={stats["steps"]}, '
                  f'compute={stats["selected_compute_count"]}, compliant={a["feasible"]}, '
                  f'cost={a["weighted_cost"]}, time={stats["wall_s"]:.2f}s, '
                  f'policy_updates={stats["parameter_updates"]}', flush=True)
            del env
        export_model(folder, recorder, protocol, model, time.perf_counter()-started)
        status.update(status='COMPLETED', phase='completed', completed_actions=committed,
                      training_completed=trainer.episodes, elapsed_s=time.perf_counter()-started)
        save_json(status_path, status)
    finally:
        recorder.close()


def human_time(seconds):
    if seconds is None: return '累積資料中'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}時{minutes:02d}分' if hours else f'{minutes}分{seconds:02d}秒'


def display_width(text):
    return sum(0 if unicodedata.combining(c) else
               2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in text)


def progress_clock(seconds):
    if seconds is None: return '--:--:--'
    hours, remainder = divmod(max(0,int(seconds)),3600)
    minutes, seconds = divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress_line(index, model, config, total, status, active, columns):
    phase = status.get('phase','validation')
    label = {'validation':'驗證','baseline':'前測','train':'訓練','periodic':'測試',
             'random':'隨機','sa':'退火','final':'後測','completed':'完成'}.get(phase,'準備')
    if active.get('stage') == 'UPDATING_POLICY': label = '更新'
    number = status.get('current_episode',0)
    limit = config['episodes'] if phase in ('train','random','sa','periodic') else 1
    steps = active.get('requests',0)
    eta = progress_clock(total['estimated_remaining_s'])
    line = (f'[{index}/{total["total_models"]}] {model:<12} {label}{number:02d}/{limit:02d}  '
            f'互動{steps:02d}/{config["steps_per_episode"]:02d}  '
            f'{100*total["progress_fraction"]:5.1f}%  '
            f'已用{progress_clock(total["elapsed_active_s"])}  剩餘{eta}')
    if display_width(line) > columns-1:
        line = (f'[{index}/{total["total_models"]}] {model} {label}{number:02d}/{limit:02d} '
                f'{steps:02d}/{config["steps_per_episode"]:02d} '
                f'{100*total["progress_fraction"]:5.1f}% ETA{eta}')
    ppa = active.get('current_ppa')
    if ppa:
        extra = (f'  P {ppa["power_w"]:.4g}W  L {1000*ppa["latency_s"]:.4g}ms'
                 f'  A {ppa["area_mm2"]:.4g}mm²')
        if display_width(line+extra) < columns: line += extra
    # Reserve the final column to prevent terminal auto-wrap. Chinese glyphs
    # occupy two columns; Python len()/ljust() alone cannot enforce this.
    clipped, width = [], 0
    for char in line:
        size = display_width(char)
        if width+size > max(0,columns-1): break
        clipped.append(char); width += size
    return ''.join(clipped)


class LiveProgress:
    """One aligned terminal row; redirected output receives summaries only."""
    def __init__(self, stream):
        self.stream = stream
        self.live = stream.isatty()
        self.active = False
        self.previous_width = 0
        self.ansi = os.name != 'nt'
        if self.live and os.name == 'nt':
            try:
                import ctypes
                import msvcrt
                handle = msvcrt.get_osfhandle(stream.fileno())
                mode = ctypes.c_ulong()
                kernel = ctypes.windll.kernel32
                # Explicit handle types prevent truncating 64-bit handles.
                kernel.GetConsoleMode.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
                kernel.SetConsoleMode.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
                self.ansi = bool(kernel.GetConsoleMode(handle,ctypes.byref(mode)) and
                                 kernel.SetConsoleMode(handle,mode.value|0x0004))
            except (OSError,ValueError,AttributeError):
                self.ansi = False

    def update(self, text, columns):
        if not self.live: return
        padding = max(0,min(self.previous_width,columns-1)-display_width(text))
        self.stream.write(('\r\033[2K' if self.ansi else '\r')+text+(' '*padding if not self.ansi else ''))
        self.stream.flush()
        self.active = True
        self.previous_width = display_width(text)

    def finish(self, message=None):
        if self.active:
            self.stream.write('\r\033[2K' if self.ansi else '\r'+' '*self.previous_width+'\r')
        if message is not None: self.stream.write(message+'\n')
        elif self.active: self.stream.write('\n')
        self.stream.flush()
        self.active = False
        self.previous_width = 0


def progress_state(run, protocol, elapsed):
    config = protocol['config']
    interval=config.get('evaluation_interval',0)
    periodic=(config['episodes']-1)//interval if interval else 0
    per_model = (config['episodes']*(1+len(config.get('comparisons',[])))+
                 (2+periodic)*len(config['evaluation_seeds']))*config['steps_per_episode']
    total, actions, train, completed = per_model*len(config['models']), 0, 0, 0
    for model in config['models']:
        s = read_json(run/model/'status.json', {})
        actions += s.get('completed_actions', 0)
        train += s.get('training_completed', 0)
        completed += s.get('status') == 'COMPLETED'
        p = read_json(run/model/'progress.json', {})
        if s.get('status') == 'RUNNING' and p.get('phase_token') == s.get('phase_token'):
            actions += p.get('requests', 0)
    actions = min(total, actions)
    eta = elapsed*(total-actions)/actions if actions else None
    return {'completed_actions': actions, 'total_actions': total, 'completed_training_episodes': train,
            'total_training_episodes': config['episodes']*len(config['models']),
            'completed_models': completed, 'total_models': len(config['models']),
            'progress_fraction': actions/total, 'elapsed_active_s': elapsed, 'estimated_remaining_s': eta}


def stop_process(process):
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_suite(run, protocol):
    from .suite_report import export_suite
    previous = read_json(run/'status.json', {})
    elapsed_before = previous.get('elapsed_active_s', 0)
    started = time.perf_counter()
    process = None
    model_started = None
    attempts = read_json(run/'model_times.json', [])
    def record_attempt(model, outcome):
        nonlocal model_started
        if model_started is None: return
        attempts.append({'model':model, 'outcome':outcome, 'finished_at_utc':timestamp(),
                         'wall_s':time.perf_counter()-model_started})
        save_json(run/'model_times.json',attempts)
        model_started = None
    print(f'輸出：{run}\n固定 REINFORCE；每模型 {protocol["config"]["episodes"]} 回合；'
          f'輸入 {protocol["config"]["input_size"]}；scope=2；顆數由 RL 選 1–16。')
    print('凍結測試不更新參數；選用的隨機／退火比較各使用與訓練相同的互動預算。')
    print('所有額外測試／比較都計入進度與時間；一次僅執行一個模型。')
    print('ETA 依已跑速度暫估，跨模型、切分增長與策略更新可能讓預估變動。Ctrl+C 可停止並保留完整回合。')
    with (run/'terminal.log').open('a', encoding='utf-8') as terminal_log:
        screen = LiveProgress(sys.stdout)
        try:
            for index, model in enumerate(protocol['config']['models'], 1):
                if read_json(run/model/'status.json', {}).get('status') == 'COMPLETED': continue
                folder = run/model
                folder.mkdir(exist_ok=True)
                with (folder/'execution.log').open('ab') as log:
                    model_started = time.perf_counter()
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'race_rl.model_suite',
                        '--worker', '--run', str(run), '--model', model], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1'))
                    last_log, last_key = 0.0, None
                    while process.poll() is None:
                        elapsed = elapsed_before+time.perf_counter()-started
                        total = progress_state(run, protocol, elapsed)
                        s, p = read_json(folder/'status.json', {}), read_json(folder/'progress.json', {})
                        active = p if p.get('phase_token') == s.get('phase_token') else {}
                        phase = s.get('phase', 'validation')
                        number = s.get('current_episode', 0)
                        steps = active.get('requests', 0)
                        key = (phase, number)
                        text = (f'[{index}/{total["total_models"]}] {model} {phase} '
                                f'{number}/{protocol["config"]["episodes"] if phase == "train" else 1} '
                                f'| action {steps}/{protocol["config"]["steps_per_episode"]} '
                                f'| 回合 {total["completed_training_episodes"]}/{total["total_training_episodes"]} '
                                f'| {100*total["progress_fraction"]:.1f}% '
                                f'| 已用 {human_time(elapsed)} | 剩餘約 {human_time(total["estimated_remaining_s"])}')
                        total.update(status='RUNNING', model=model, phase=phase, current_stage=active.get('stage'))
                        save_json(run/'status.json', total)
                        columns = shutil.get_terminal_size(fallback=(100,24)).columns
                        screen.update(progress_line(index,model,protocol['config'],total,s,active,columns),columns)
                        if time.monotonic()-last_log >= 30 or key != last_key:
                            terminal_log.write(timestamp()+' '+text+'\n'); terminal_log.flush()
                            last_log, last_key = time.monotonic(), key
                        time.sleep(1)
                state = read_json(folder/'status.json', {})
                if process.returncode:
                    state.update(status='FAILED')
                    save_json(folder/'status.json', state)
                    screen.finish(f'[{index}/{len(protocol["config"]["models"])}] {model:<12} 失敗；保留已完成回合。紀錄：{folder / "execution.log"}')
                else:
                    screen.finish(f'[{index}/{len(protocol["config"]["models"])}] {model:<12} 完成  '
                                  f'{state.get("training_completed",0)}/{protocol["config"]["episodes"]} 回合  '
                                  f'耗時 {progress_clock(time.perf_counter()-model_started)}')
                record_attempt(model,'COMPLETED' if process.returncode == 0 else 'FAILED')
                export_suite(run, protocol)
            failures = [m for m in protocol['config']['models']
                        if read_json(run/m/'status.json', {}).get('status') != 'COMPLETED']
            status = progress_state(run, protocol, elapsed_before+time.perf_counter()-started)
            status['status'] = 'PARTIAL' if failures else 'COMPLETED'
            status['failed_models'] = failures
            save_json(run/'status.json', status)
            export_suite(run, protocol)
            print(f'批次狀態：{status["status"]}；報告：{run / "report.html"}', flush=True)
            return 1 if failures else 0
        except KeyboardInterrupt:
            screen.finish()
            stop_process(process)
            # The worker checkpoint/records commit only at episode boundaries.
            if process is not None and model_started is not None:
                s = read_json(run/model/'status.json', {})
                s.update(status='INTERRUPTED')
                save_json(run/model/'status.json', s)
                record_attempt(model,'INTERRUPTED')
            status = progress_state(run, protocol, elapsed_before+time.perf_counter()-started)
            status['status'] = 'INTERRUPTED'
            save_json(run/'status.json', status)
            export_suite(run, protocol)
            print(f'\n已停止。從最後完整回合繼續：\n{sys.executable} tools/train_model_suite.py --legacy-synthetic --resume {run}')
            return 130
        finally:
            screen.finish()
            stop_process(process)


def main(argv=None):
    from .frontend import SUPPORTED_MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', type=Path, help='廠商限制；顯式指定時覆蓋研究任務，不自動放寬')
    parser.add_argument('--tasks',type=Path,default=ROOT/'configs/model_suite_tasks.json',help='各模型具量測可行設計的研究限制')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--evaluation-seeds', type=int, nargs='+', default=list(range(1007,1017)))
    parser.add_argument('--evaluation-interval',type=int,default=0,help='選用中途凍結測試；0 表示只做前後測')
    parser.add_argument('--comparisons',nargs='*',choices=('random','sa'),default=[],help='選用額外等預算比較；每種增加 3840 次互動／模型')
    parser.add_argument('--reward',choices=('dense','terminal'),default='dense',help='預先指定 reward 消融；不宣稱 terminal 較好')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--entropy-coefficient',type=float,default=0.0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke', action='store_true', help='all seven models, 1 episode/2 actions; verification only')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--model', choices=SUPPORTED_MODELS, help='進階：只跑指定模型；預設依序訓練全部七個模型')
    args = parser.parse_args(argv)
    if args.worker:
        try: worker(args.run, args.model)
        except BaseException:
            traceback.print_exc()
            sys.exit(1)
        return
    if args.resume:
        run = args.resume.resolve()
        protocol = read_json(run/'protocol.json')
        if not protocol or protocol['source_hashes'] != code_hashes():
            parser.error('缺少 protocol 或來源已變更，不能接續此實驗。')
    else:
        from .vendor_input import load_requirements
        from .feasibility import load_tasks
        selected_models=[args.model] if args.model else list(SUPPORTED_MODELS)
        tasks=load_tasks(args.tasks,selected_models) if args.requirements is None else {}
        req=load_requirements(args.requirements) if args.requirements else None
        if args.evaluation_interval<0 or len(set(args.comparisons))!=len(args.comparisons):
            parser.error('測試間隔需非負；比較方法不可重複')
        import math
        if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.entropy_coefficient) or args.entropy_coefficient<0:
            parser.error('learning rate must be positive and finite; entropy coefficient nonnegative and finite')
        if not 0 <= args.seed <= 2**31-1 or any(not 0 <= s <= 2**31-1 for s in args.evaluation_seeds) or len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
            parser.error('seeds must be distinct nonnegative integers')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        base = ROOT/'experiments/runs/model_suite'/('smoke' if args.smoke else 'training')
        if not args.smoke and args.model: base = base/args.model
        run = (args.out or base/stamp).resolve()
        if run.exists() and any(run.iterdir()): parser.error('輸出目錄必須為空，接續請用 --resume。')
        run.mkdir(parents=True, exist_ok=True)
        config = {'models': selected_models, 'episodes': 1 if args.smoke else 60,
                  'steps_per_episode': 2 if args.smoke else 64, 'input_size': 64,
                  'seed': args.seed, 'evaluation_seeds': args.evaluation_seeds,
                  'requirements': plain(req), 'tasks':tasks,'reward_variant':args.reward,
                  'learning_rate':args.learning_rate,'entropy_coefficient':args.entropy_coefficient,
                  'evaluation_interval':0 if args.smoke else args.evaluation_interval,
                  'comparisons':args.comparisons,'max_scope': 2, 'compute_counts': COUNTS,
                  'algorithm': 'reinforce', 'precision': 'float64', 'early_stopping': False}
        from .experiment_diagnostics import platform_snapshot
        protocol = {'schema': 2, 'started_at_utc': timestamp(), 'config': config,
                    'platform_at_start':platform_snapshot(),
                    'task_source':'Measured pilot-derived synthetic feasibility tasks' if tasks else 'Explicit vendor requirements',
                    'fairness':'Same complete-action REQUEST budget and resets per comparison; new PPA evaluations and cache hits reported separately. Test rollouts are additional, not part of training.',
                    'source_hashes': code_hashes(), 'smoke_only': args.smoke,
                    'claims': 'same-workload learning diagnostic; synthetic uncalibrated PPA; '
                              'one training seed; paired frozen policy tests, not generalization proof',
                    'recording': 'compressed per-action feedback/design-state in SQLite; '
                                 'one latest transactional checkpoint; one full selected-design ledger',
                    'identity': digest({'config': config, 'source': code_hashes()})}
        save_json(run/'protocol.json', protocol)
    sys.exit(run_suite(run, protocol))


if __name__ == '__main__':
    main()
