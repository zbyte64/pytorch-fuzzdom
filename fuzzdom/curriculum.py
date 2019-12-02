import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
import random
from collections import deque, defaultdict, Counter

from .domx import embed, short_embed


class TaskCompetency:
    def __init__(self, n_tasks, sample_size=100):
        self.n_tasks = n_tasks
        self.task_history = [deque(maxlen=sample_size) for i in range(n_tasks)]

    def forward(self, task_ids):
        tp = []
        for task_id in task_ids:
            c = Counter(self.task_history[task_id])
            success = c[True]
            fail = c[False]
            if not success and not fail:
                accuracy = 0.0
            else:
                accuracy = success / (success + fail)
            tp.append(accuracy)
        tp = torch.tensor(tp).view(1, -1)
        return Bernoulli(probs=tp)

    def difficulty_rank(self):
        if not hasattr(self, "_difficulty_rank"):
            self.compute_difficulty_rank()
        return self._difficulty_rank

    def compute_difficulty_rank(self):
        with torch.no_grad():
            task_idx = torch.arange(0, self.n_tasks, dtype=torch.long)
            d = self.forward(task_idx)
            self._level_probs = d.probs.view(1, -1)
            p = d.probs.tolist()
            ranks = list(zip(p, task_idx.tolist()))
            ranks.sort(key=lambda x: x[0], reverse=True)
        self._difficulty_rank = ranks

    def update(self):
        self.compute_difficulty_rank()
        return 0.0

    def record(self, task, pass_or_fail, task_fields):
        self.task_history[task].append(bool(pass_or_fail))


class LevelTracker:
    global_scoreboard = {}
    running_levels = defaultdict(lambda: 0)

    def __init__(self, levels, predictor, num_processes, K=2):
        #TODO support a list of list of levels, 0-dim represents task type, 1-dim difficulty
        self.predictor = predictor
        self.levels = levels
        self.num_processes = num_processes
        self.K = min(K, num_processes)
        self._k = _k = hash(tuple(levels))
        if _k in self.global_scoreboard:
            self.level_stats = self.global_scoreboard[_k]
        else:
            self.level_stats = {
                i: {"pass": 0, "fail": 0, "id": i, "level": l}
                for i, l in enumerate(self.levels)
            }
            self.global_scoreboard[_k] = self.level_stats
        self._train_data = ([], [])
        self.past_top_k = deque(maxlen=20)
        self.select_level()

    def rank_levels(self):
        rankings = self.predictor.difficulty_rank()
        self.ranked_levels = list()
        for pass_prob, level_idx in rankings:
            self.ranked_levels.append(self.level_stats[level_idx])

    def get_level_stat(self):
        return self.level_stats[self.current_level]

    def get_level(self):
        next_level = self.get_level_stat()
        return next_level["level"]

    def select_level(self):
        if hasattr(self, "current_level"):
            self.running_levels[self.current_level] -= 1
        self.rank_levels()
        probs = self.predictor._level_probs.clone().detach()

        # sample other levels
        if self.running_levels:
            min_run = min(*self.running_levels.values())
        else:
            min_run = 0
        not_running = list(
            filter(lambda x: self.running_levels[x] == min_run, range(len(self.levels))),
        )
        self.current_level = random.choice(not_running)
        self.task_runs = 10
        # self.current_level = max(min(self.current_level, len(self.levels) - 1), 0)
        self.running_levels[self.current_level] += 1

    def __call__(self, pass_or_fail, task_fields):
        stats = self.get_level_stat()
        self.predictor.record(stats["id"], pass_or_fail, task_fields)
        self.task_runs -= 1
        if pass_or_fail:
            stats["pass"] += 1
        else:
            stats["fail"] += 1
        if self.task_runs < 0:
            self.select_level()
        return self.get_level()


if __name__ == "__main__":
    t = TaskSkillCompetency(40, 6)
    # _s1 = [t.skill_proficiencies.tolist(), t.task_skills.tolist()]
    for i in range(40):
        t.record(i, True)
        t.record(i, False)
        if i > 10:
            t.record(i, False)
    assert t.update() > 0
    # assert _s1 != [t.skill_proficiencies.tolist(), t.task_skills.tolist()]
