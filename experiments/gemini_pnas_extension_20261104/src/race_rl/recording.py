"""Compact, transactional experiment records; never called to choose an action."""
import io
import json
import sqlite3
import zlib
from pathlib import Path
from .serialization import plain, digest


def packed(value):
    return zlib.compress(json.dumps(plain(value), allow_nan=False, separators=(',', ':')).encode(), 6)


def unpacked(value):
    return json.loads(zlib.decompress(value))


def policy_hash(policy):
    import hashlib
    sha = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        sha.update(name.encode())
        sha.update(value.detach().numpy().tobytes())
    return sha.hexdigest()


def assessment_better(a, b):
    if not a['physical_feasible']: return False
    if b is None or not b['physical_feasible']: return True
    if a['search_merit'] != b['search_merit']:
        return a['search_merit'] < b['search_merit']
    from .requirements import METRICS
    x, y = tuple(a['metrics'][k] for k in METRICS), tuple(b['metrics'][k] for k in METRICS)
    return all(v <= w for v, w in zip(x, y)) and any(v < w for v, w in zip(x, y))


def interaction_records(env):
    """Keep actual feedback and replayable design state; omit duplicate ledgers."""
    state = {'hardware_hash': digest(env.hardware_candidates[env.initial_count]),
             'mapping_hash': digest(env.initial), 'compute_count': env.initial_count,
             'objective': 1.0, 'ppa': None, 'last_reward': 0.0}
    rows = []
    for index, trace in enumerate(env.trace, 1):
        next_state = dict(state)
        if trace['physical_feasible'] and trace.get('controller_accepted',True):
            next_state.update(hardware_hash=trace['candidate_hardware_hash'],
                              mapping_hash=trace['candidate_mapping_hash'],
                              ppa=trace['ppa_assessment'])
        next_state.update(compute_count=trace['compute_count'], objective=trace['current_objective'],
                          last_reward=trace['reward'])
        rows.append({**plain(trace), 'step': index, 'state_before': state,
                     'state_after': next_state,
                     'remaining_evaluations': env.oracle.budget-trace['evaluations'],
                     'remaining_requests': env.settings['max_requests']-trace['requests']})
        state = next_state
    return rows


class CompactRecorder:
    """One SQLite file per model; completed episodes and checkpoint commit together."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
              phase TEXT NOT NULL, seed INTEGER NOT NULL, episode INTEGER NOT NULL,
              stats TEXT NOT NULL, interactions BLOB NOT NULL,
              PRIMARY KEY(phase,seed,episode));
        ''')
        previous = self.get('identity')
        if previous is not None and previous != identity:
            self.close()
            raise ValueError('record identity differs; cannot resume a different experiment')
        with self.db:
            self.put('identity', identity)

    def put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, packed(value)))

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return unpacked(row[0]) if row else default

    def sessions(self):
        return [{'phase': p, 'seed': s, 'episode_number': n, **json.loads(v)}
                for p, s, n, v in self.db.execute(
                    'SELECT phase,seed,episode,stats FROM episodes ORDER BY rowid')]

    def has(self, phase, seed, episode):
        return self.db.execute('SELECT 1 FROM episodes WHERE phase=? AND seed=? AND episode=?',
                               (phase, seed, episode)).fetchone() is not None

    def checkpoint(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='checkpoint'").fetchone()
        return io.BytesIO(row[0]) if row else None

    def record(self, phase, seed, episode, env, stats, *, trainer=None, retain_winner=False):
        if env.oracle.requests != len(env.trace):
            raise ValueError('interaction accounting mismatch')
        winner_key='winner' if phase=='train' else f'winner_{phase}'
        before = self.get(winner_key+'_assessment') if retain_winner else None
        assessment = env.scorer.assess(env.best_record)
        winner = None
        if retain_winner and assessment_better(assessment, before):
            winner = {'assessment': assessment, 'hardware': plain(env.best_hardware),
                      'mapping': plain(env.best_mapping), 'evaluation': plain(env.best_record),
                      'episode_number': episode}
        checkpoint = None
        if trainer is not None:
            buffer = io.BytesIO()
            trainer.save(buffer, experiment_identity=self.get('identity'))
            checkpoint = buffer.getvalue()
        # A duplicate key or interrupted transaction cannot add an episode
        # without its matching policy/RNG checkpoint.
        with self.db:
            self.db.execute('INSERT INTO episodes VALUES (?,?,?,?,?)',
                            (phase, seed, episode, json.dumps(plain(stats), allow_nan=False),
                             packed(interaction_records(env))))
            if checkpoint is not None:
                self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('checkpoint', checkpoint))
            if winner is not None:
                self.put(winner_key, winner)
                self.put(winner_key+'_assessment', assessment)

    def close(self):
        self.db.close()
