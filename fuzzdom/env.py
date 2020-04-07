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
import asyncio

from miniwob.reward import get_original_reward, get_raw_reward
from miniwob.fields import Fields

from PIL import Image

from .state import MiniWoBGraphState, fields_factory
from .domx import json_to_graph, miniwob_to_graph
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


async def open_driver(proxy=None):
    chromeOptions = {
        "args": [
            "enable-automation",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--disable-browser-side-navigation",
        ]
    }
    if proxy and False:
        # TODO arsenic proxy?
        prox = Proxy()
        prox.proxy_type = ProxyType.MANUAL
        prox.http_proxy = proxy
        prox.socks_proxy = proxy
        prox.ssl_proxy = proxy

        prox.add_to_capabilities(capabilities)

    if "SELENIUM_URL" in os.environ:
        driver = await start_session(
            services.Remote(os.getenv("SELENIUM_URL")),
            browsers.Chrome(chromeOptions=chromeOptions),
        )
    else:
        driver = await start_session(
            services.Chromedriver(), browsers.Chrome(chromeOptions=chromeOptions)
        )
    return driver


class WebInterface:
    """
    Interface to interact with a browser instance
    """

    def __init__(self, driver=None):
        self._driver = driver
        self._load_code = "".join(random.sample(string.ascii_letters, 8))

    async def open(self):
        print("Browser Task: " + str(self._driver))

    async def close(self):
        pass

    async def _injection_check(self):
        # inject javascript here
        loaded = await self._driver.execute_script(
            f"document.body.dataset.{self._load_code}"
        )
        if not loaded:
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
    def __init__(self, proxy=None):
        super(ManagedWebInterface, self).__init__()
        self.proxy = proxy

    async def open(self):
        self._driver = await open_driver(self.proxy)
        return await super(ManagedWebInterface, self).open()

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
    if ref is not None:
        knows_ref = await device.execute_script(
            f"return core.previousDOMInfo[{ref}] != null"
        )
        if knows_ref:
            await device.execute_script(
                f"""
                core.previousDOMInfo['{ref}'].click();
                core.previousDOMInfo['{ref}'].focus();
                core.previousDOMInfo['{ref}'].value = '{text}';
                """
            )
            return
        else:
            pass
            # print("previous dom info does not have ref:", ref, knows_ref)
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
        base_url=os.getenv("CRAWL_PATH"),
        wait_ms=0.0,
        level_tracker=None,
        web_interface=None,
    ):
        super(MiniWoBGraphEnvironment, self).__init__()
        self.wait_ms = wait_ms
        self._base_url = base_url
        self._levels = levels
        self._web = web_interface
        self._actions = [
            lambda ref, value: (
                ref
                and self.driver.execute_script(f"return core.elementClick({ref});")
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

    async def wob_dom(self) -> nx.DiGraph:
        dom_info = await self.run_script("return core.getDOMInfo();")
        if "ref" in dom_info:
            return miniwob_to_graph(dom_info)
        exclude_ids = {"reward-display", "sync-task-cover", "click-canvas", "query"}
        return json_to_graph(
            dom_info, exclude=lambda n: n["a"].get("id") in exclude_ids
        )

    async def _injection_check(self):
        location = await self._web.location
        assert location["protocol"] != "chrome-error:"

    async def refresh_state(self):
        # updates with new state with refetched dom and image
        await self.wait_for_dom()
        await self._injection_check()
        self.state = MiniWoBGraphState(
            self.state.utterance,
            self.state.fields,
            await self.wob_dom(),
            None,  # await self._web.get_img(),
        )

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
            self._web = ManagedWebInterface(proxy=os.getenv("PROXY_HOST"))
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
        assert ref in self.state.dom_graph, str(ref)
        f = self._actions[action_id]

        start_time = time.time()
        waitable = f(ref, value)
        if waitable is not None:
            try:
                await waitable
            except:
                dom_g = await self.wob_dom()
                if ref not in dom_g:
                    print("Bad node?", ref, self.state.dom_graph.nodes[ref])
                else:
                    print("Node was found!", ref)
                raise
        # wait for an amount of time but take into account future js execution time
        wait_time = max(self.wait_ms / 1000 - (time.time() - start_time) * 2, 0)
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
        if hasattr(self, "level_tracker"):
            print("Level scoreboard", self.level_tracker.ranked_levels)
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

    async def get_wob_state(self) -> MiniWoBGraphState:
        # Get the utterance
        response = await self.run_script("return core.getUtterance();")
        fields = fields_factory(self.task, response)
        # Get the DOM
        dom_graph = await self.wob_dom()
        img = None  # await self._web.get_img()
        state = MiniWoBGraphState(response, fields, dom_graph, img)
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

    def __init__(self, driver, wait_ms=0.0):
        web_interface = WebInterface(driver=driver)
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
        dom_graph = await self.wob_dom()
        state = MiniWoBGraphState(utterance, fields, dom_graph, None)
        return state

    async def wob_dom(self) -> nx.DiGraph:
        dom_info = await self.run_script("return core.getDOMInfo();")
        return json_to_graph(dom_info)

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

    async def _injection_check(self):
        await self._web._injection_check()
