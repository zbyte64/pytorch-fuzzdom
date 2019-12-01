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
        ranks = list(
            filter(
                lambda x: x[0] < 0.85 and x[0] > 0.2 and self.level_stats[x[1]]["fail"],
                zip(probs.view(-1).tolist(), range(0, probs.shape[1])),
            )
        )
        ranks.sort(key=lambda x: x[0], reverse=True)

        # train on top K levels (and those seen in the past)
        for prob, idx in ranks[: self.K]:
            if self.level_stats[idx]["pass"] > 4:
                self.past_top_k.append(idx)
        top_counts = Counter(self.past_top_k)
        train_targets = [idx for idx, c in top_counts.most_common(self.K)]
        C = max(self.num_processes // self.K - 1, 0)
        running = set(
            map(lambda x: x[0], filter(lambda x: x[1] > C, self.running_levels.items()))
        )
        choices = [idx for idx in train_targets if idx not in running]
        if choices:
            self.current_level = random.choice(choices)
            self.task_runs = 100
        else:
            # sample other levels
            # a single mountain peak to favor the not-yet mastered
            running = list(
                map(
                    lambda x: x[0],
                    filter(lambda x: x[1] > 0, self.running_levels.items()),
                )
            )
            peak = 0.75
            mastered_mask = probs > peak
            running_mask = torch.tensor(running, dtype=torch.long)
            probs[0, running_mask] = 0
            probs[mastered_mask] *= 1 - (probs[mastered_mask] - peak) / (1 - peak)
            probs += 1e-4
            dist = Categorical(logits=probs)
            self.current_level = dist.sample().item()
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
