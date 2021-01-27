import os
import json
import time
import numpy as np
import io
import gym
from gym import spaces
import random
import string
from functools import lru_cache
import logging
import networkx as nx

from arsenic import (
    get_session,
    keys,
    browsers,
    services,
    start_session,
    stop_session,
    actions,
)
from arsenic.errors import ArsenicError
import asyncio

from miniwob.reward import get_original_reward, get_raw_reward
from miniwob.fields import Fields

from PIL import Image

from .state import MiniWoBGraphState, fields_factory, DomInfo
from .domx import miniwob_to_dominfo
from .dir_paths import JS_PATH


# https://github.com/HDE/arsenic/issues/35
def set_arsenic_log_level(level=logging.WARNING):
    # Create logger
    logger = logging.getLogger("arsenic")

    # We need factory, to return application-wide logger
    def logger_factory():
        return logger

    import structlog

    structlog.configure(logger_factory=logger_factory)
    logger.setLevel(level)


set_arsenic_log_level()


async def open_driver(capabilities={"browserName": "chrome"}):
    if "SELENIUM_URL" in os.environ:
        driver = await start_session(
            services.Remote(os.getenv("SELENIUM_URL")), browsers.Browser(**capabilities)
        )
    else:
        driver = await start_session(
            services.Chromedriver(), browsers.Browser(**capabilities)
        )
    return driver


class WebInterface:
    """
    Interface to interact with a browser instance
    """

    def __init__(self, driver=None):
        self._driver = driver
        self._load_code = None

    async def _injection_check(self):
        loaded = False
        if self._load_code:
            loaded = (
                await self._driver.execute_script(
                    f"return document.body.dataset.{self._load_code}"
                )
                is True
            )
        else:
            self._load_code = "".join(random.sample(string.ascii_letters, 8))
        if not loaded:
            # inject javascript here
            await self._driver.execute_script(self.initial_js_code())
            await self._driver.execute_script(
                f"document.body.dataset.{self._load_code} = true;"
            )

    async def wait_for_dom(self):
        while (
            await self._driver.execute_script("return document.readyState;")
            != "complete"
        ):
            await asyncio.sleep(0.1)

    @lru_cache()
    def initial_js_code(self):
        return open(os.path.join(JS_PATH, "init.js"), "r").read()

    @lru_cache()
    def js_get_visible_dom(self):
        return open(os.path.join(JS_PATH, "get_visible_dom.js"), "r").read()

    @property
    def html(self):
        return self._driver.execute_script("return document.body.innerHTML;")

    @property
    def visible_dom(self):
        return self._driver.execute_script(self.js_get_visible_dom())

    @property
    def location(self):
        return self._driver.execute_script("return window.location;")

    @property
    def scroll_position(self):
        return self._driver.execute_script("return [window.scrollX, window.scrollY];")

    @property
    def mouse_position(self):
        return self._driver.execute_script(
            "return [window.clientX || 0, window.clientY || 0];"
        )

    async def get_img(self):
        png_data = await self._driver.get_screenshot()
        pil_img = Image.open(png_data)
        return pil_img


class ManagedWebInterface(WebInterface):
    """
    Provides a driver from the environment with a built-in circuit-breaker
    """

    def __init__(self, error_circuit_break=3):
        super().__init__()
        self._error_count = 0
        self._error_circuit_break = error_circuit_break

    async def open(self):
        try:
            self._driver = await open_driver()
            await self._injection_check()
        except ArsenicError:
            self._error_count += 1
            if self._error_count > self._error_circuit_break:
                raise RunTimeError("Repetitive driver errors caused circuit break")
            raise
        else:
            self._error_count = 0
        print("Browser Task: " + str(self._driver))

    async def close(self):
        if getattr(self, "_driver", None):
            print("Closing Chrome")
            try:
                await stop_session(self._driver)
            except Exception as e:
                print(e)
                pass
            self._driver = None


async def send_keys(device, text: str, ref=None):
    assert isinstance(text, str), str(text)
    if ref is not None:
        knows_ref = await device.execute_script(
            f"""
            return core.elementType('{ref}', '{text}');
            """
        )
        return knows_ref
    ticks = []
    keyboard = actions.Keyboard("keyboard")
    for key in text:
        keyCode = 0
        charCode = ord(key)
        await device.execute_script(
            f"""
var keyboardEvent = new KeyboardEvent("keypress", {{bubbles:true, key:'{key}'}})
document.dispatchEvent(keyboardEvent);"""
        )


class MiniWoBGraphEnvironment(gym.Env):
    """
    Gym environment for intepreting MiniWoB as a graph.
    """

    observation_space = spaces.Discrete(2 * 31)
    reward_range = (0.0, 1.0)
    metadata = {"render.modes": ["graph"]}
    select_level_lock = asyncio.Lock()

    def __init__(
        self,
        levels,
        web_interface=None,
        base_url=os.getenv("CRAWL_PATH"),
        wait_ms=0.0,
        level_tracker=None,
    ):
        super(MiniWoBGraphEnvironment, self).__init__()
        self.wait_ms = wait_ms
        self._base_url = base_url
        self._levels = levels
        self._web = web_interface
        self._actions = [
            lambda ref, value: (
                ref
                and self.driver.execute_script(f"return core.elementClick('{ref}');")
                or None
            ),
            lambda ref, value: send_keys(self.driver, value, ref),
            lambda ref, value: self.set_state(self.state.copy_node_text(ref)),
            lambda ref, value: send_keys(self.driver, self.state.clipboard_text, ref),
            lambda ref, value: asyncio.sleep(0.5),
        ]
        self._action_meanings = [
            "Click Node",
            "Paste Field Text",
            "Copy Node Text",
            "Paste Clipboard",
            "Wait",
        ]
        self.action_space = spaces.Discrete(len(self._actions))
        self.state = None
        self.level_tracker = level_tracker

    @property
    def driver(self):
        return self._web._driver

    async def run_script(self, s: str):
        return await self.driver.execute_script(s)

    async def wob_dom(self) -> DomInfo:
        dom_info = await self.run_script("return core.getDOMInfo();")
        if "col" not in dom_info:
            dom_info = miniwob_to_dominfo(dom_info)
        return DomInfo(**dom_info)

    async def _injection_check(self):
        location = await self._web.location
        assert location["protocol"] != "chrome-error:"
        await self._web._injection_check()

    async def get_js_logs(self):
        return await self.run_script("return core.getLogs();")

    async def refresh_state(self):
        # updates with new state with refetched dom and image
        await self.wait_for_dom()
        await self._injection_check()
        self.state = await self.get_wob_state()

    def set_state(self, state):
        assert state
        self.state = state

    async def wait_for_dom(self):
        await self._web.wait_for_dom()

    def reset(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_reset())

    async def async_reset(self) -> MiniWoBGraphState:
        if not self._web:
            self._web = ManagedWebInterface()
        if not self.driver:
            await self.open()
        await self.select_level_lock
        self.level_tracker.select_level()
        self.select_level_lock.release()
        await self.begin_task()
        return self.state

    async def render(self):
        return self.state

    def step(self, action) -> tuple:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_step(action))

    async def async_step(self, action) -> tuple:
        action_id, ref, value = action
        assert isinstance(value, str), str(value)
        # assert ref in self.state.dom_info.nodes, str(ref)
        f = self._actions[action_id]
        # print("#", action_id, ref, value)

        start_time = time.time()
        waitable = f(ref, value)
        if waitable is not None:
            try:
                await waitable
            except:
                # dom_g = await self.wob_dom()
                # if ref not in dom_g.nodes:
                #    print("Bad node?", ref, self.state.dom_info.nodes[ref])
                # else:
                #    print("Node was found!", ref)
                raise
        # wait for an amount of time but take into account future js execution time
        wait_time = 0  # max(self.wait_ms / 1000 - (time.time() - start_time) * 2, 0)
        if wait_time:
            await asyncio.sleep(wait_time)
        await self.refresh_state()
        metadata = await self.get_metadata()
        r = get_raw_reward(metadata)
        done = False
        task_done = metadata["done"]
        metadata["task_done"] = task_done
        metadata["task"] = self.task
        task_success = r > 0
        if task_done:
            print(task_success, self.task, r)
            task_description = self.state.utterance
            next_level = self.level_tracker(task_success, self.state.fields)
            await self.begin_task()
            metadata["episode"] = {"r": r}
        else:
            # wait for any remaining amount of time
            wait_time = max(self.wait_ms / 1000 - time.time() - start_time, 0)
            if wait_time:
                await asyncio.sleep(wait_time)
        metadata["elapsed"] = max(0.0, time.time() - self.start_time)
        return self.state, r, done, metadata

    async def open(self):
        await self._web.open()

    async def close(self):
        await self._web.close()

    def get_action_meanings(self):
        return self._action_meanings

    async def force_stop(self):
        await self.run_script("return core.endEpisode(0);")

    async def begin_task(self, seed: float = None):
        entry_url = self.level_tracker.get_level()
        self.task = entry_url.split("/")[-1].rsplit(".", 2)[0]
        url = self._base_url + entry_url
        await self.driver.get(url)
        await self.wait_for_dom()
        await self._injection_check()

        if seed is not None:
            await self.set_seed(seed)
        try:
            await self.set_mode("train")
        except:
            l = await self._web.location
            print("Exception raised from:", l["href"])
            raise
        await self.run_script("core.startEpisodeReal();")
        self.start_time = time.time()
        self.state = await self.get_wob_state()
        # print((self.task, self.state.fields))

    async def get_wob_state(self) -> MiniWoBGraphState:
        # Get the utterance
        response = await self.run_script("return core.getUtterance();")
        fields = fields_factory(self.task, response)
        # Get the DOM
        dom_info = await self.wob_dom()
        img = None  # await self._web.get_img()
        logs = {"errors": [], "log": []}
        state = MiniWoBGraphState(response, fields, dom_info, img, logs)
        return state

    async def get_metadata(self) -> dict:
        return await self.run_script(
            "return {"
            '"done": WOB_DONE_GLOBAL,'
            '"env_reward": WOB_REWARD_GLOBAL,'
            '"raw_reward": WOB_RAW_REWARD_GLOBAL,'
            '"reason": WOB_REWARD_REASON,'
            "};"
        )

    async def set_seed(self, seed: float):
        await self.run_script("Math.seedrandom({});".format(repr(seed)))

    async def set_mode(self, mode: str):
        await self.run_script('core.setDataMode("{}");'.format(mode))


class CustomTaskEnvironment(MiniWoBGraphEnvironment):
    """
    Modified Graph Environment for executing custom commands
    Injects DOM serialization javascript into browser session

    env = CustomTaskEnvironment(driver)
    env.set_task("Click on the Movies tab", [("click", "Movies")])
    """

    def __init__(self, driver=None, wait_ms=0.0):
        web_interface = WebInterface(driver=driver) if driver else None
        super(CustomTaskEnvironment, self).__init__(
            levels=None,
            base_url="",
            wait_ms=wait_ms,
            level_tracker=None,
            web_interface=web_interface,
        )
        self.task = "custom task"

    def set_task(self, task_fields: dict, utterance: str):
        self.fields = Fields(task_fields)
        self.utterance = utterance

    async def async_reset(self):
        if not self._web:
            self._web = ManagedWebInterface(proxy=os.getenv("PROXY_HOST"))
        if not self.driver:
            await self.open()
        try:
            await self.begin_task()
        except ArsenicError as error:
            print("Error while beginning task")
            print(error)
            await self.close()
            # TODO random time interval
            await asyncio.sleep(0.1)
            await self.open()
            await self.begin_task()
        return self.state

    async def begin_task(self, seed=None):
        await self.wait_for_dom()
        await self._web._injection_check()
        self.start_time = time.time()
        self.state = await self.get_wob_state()

    async def get_wob_state(self) -> MiniWoBGraphState:
        utterance = self.utterance
        fields = self.fields
        # Get the DOM
        dom_info = await self.wob_dom()
        logs = await self.get_js_logs()
        if any(logs.values()):
            print("JS logs:")
            print(logs)
        state = MiniWoBGraphState(utterance, fields, dom_info, None, logs)
        return state

    async def wob_dom(self) -> DomInfo:
        dom_info = await self.run_script("return core.getDOMInfo();")
        return DomInfo(**dom_info)

    async def get_metadata(self) -> dict:
        reward = 0.0
        reason = None
        done = False
        return {
            "done": done,
            "env_reward": reward,
            "raw_reward": reward,
            "reason": reason,
        }

    async def async_step(self, action) -> tuple:
        await self.wait_for_dom()
        await self._web._injection_check()
        return await super().async_step(action)


class CrawlTaskEnvironment(CustomTaskEnvironment):
    """
    Modified Graph Environment for crawling a site

    env = CrawlTaskEnvironment(driver, start_url)
    """

    def __init__(self, start_url, wait_ms=0.0, valid_url=lambda href: True):
        super().__init__(wait_ms=wait_ms)
        self.start_url = start_url
        self.valid_url = valid_url
        self.set_task({"click": ""}, "Explore")

    async def begin_task(self, seed=None):
        await self.driver.get(self.start_url)
        await self.wait_for_dom()
        await self._web._injection_check()
        await self.wait_for_dom()
        self.start_time = time.time()
        self.state = await self.get_wob_state()

    async def get_metadata(self) -> dict:
        l = await self._web.location
        reward = 0.0
        reason = None
        done = not self.valid_url(l["href"])
        return {
            "done": done,
            "env_reward": reward,
            "raw_reward": reward,
            "reason": reason,
            "href": l["href"],
            "episode": {"r": reward},
        }

    async def async_reset(self):
        if not self._web:
            self._web = ManagedWebInterface()
        while True:
            try:
                if not self.driver:
                    await self.open()
                await self.begin_task()
            except ArsenicError as error:
                print("Error while beginning task")
                print(error)
                await self.close()
                await asyncio.sleep(random.random())
            else:
                return self.state
