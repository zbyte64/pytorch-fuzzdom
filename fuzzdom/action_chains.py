import asyncio
import torch
import os
from collections import OrderedDict, Counter

from .env import CustomTaskEnvironment
from .vec_env import make_vec_envs
from .storage import StorageReceipt
from .dir_paths import PRETRAINED_PATH


class FuzzyActionChains(object):
    """
    actions = FuzzyActionChains(driver, "Enter username and password and login")
    actions.input_field("username", "BillyJane")
    actions.input_field("password", "pas$word")
    actions.submit()
    actions.perform()
    """

    def __init__(self, driver, utterance: str, agent=None, stop_condition=None):
        self.driver = driver
        if agent is None:
            agent = torch.load(os.path.join(PRETRAINED_PATH, "agent.pt"))
            agent.eval()
        self.agent = agent
        self.receipts = StorageReceipt()
        self.agent.receipts = self.receipts
        self._env = CustomTaskEnvironment(driver)
        self.env = make_vec_envs([self._env], self.receipts)
        self.utterance = utterance
        self.task_fields = []
        self.stop_condition = stop_condition

    def perform(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.async_perform())

    async def async_perform(self):
        max_steps = len(self.task_fields) * 4
        c = Counter()
        task_fields = OrderedDict()
        for k, v in self.task_fields:
            c.update([k])
            n = c[k] - 1
            if n:
                target = f"{k} {n}"
            else:
                target = k
            task_fields[target] = c
        await self.async_run_task(
            task_fields, self.utterance, max_steps, self.stop_condition
        )

    async def async_run_task(
        self, task_fields: dict, utterance: str, max_steps: int, stop_condition
    ):
        self._env.set_task(task_fields, utterance)
        await self._env.begin_task()
        masks = torch.zeros(1, 1)
        recurrent_hidden_states = torch.zeros(1, 128)
        obs = await self.env.async_reset()
        for i in range(max_steps):
            obs_v = self.receipts.redeem(torch.tensor(obs))
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = self.agent.act(obs_v, recurrent_hidden_states, masks)

            obs, infos = await self.env.async_step(action)
            if stop_condition and await stop_condition():
                return
        self.receipts.prune(torch.tensor(obs))

    def click(self, description: str):
        self.task_fields.append(("click", description))
        return self

    def select(self, description: str):
        self.task_fields.append(("select", description))
        return self

    def input_field(self, field: str, string: str):
        self.task_fields.append((field, string))
        return self

    def submit(self, description: str = "submit"):
        self.task_fields.append(("submit", description))
        return self

    def copy(self, description: str):
        self.task_fields.append(("copy", description))
        return self

    def paste(self, description: str):
        self.task_fields.append(("paste", description))
        return self
