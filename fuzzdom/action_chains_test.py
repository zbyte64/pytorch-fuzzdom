import pytest
from .action_chains import FuzzyActionChains
from .env import open_driver
from .dir_paths import MINIWOB_HTML


@pytest.mark.asyncio
async def test_action_chain():
    agent = None
    driver = await open_driver()
    url = f"file://{MINIWOB_HTML}/miniwob/login-user.html"
    await driver.get(url)
    actions = FuzzyActionChains(
        driver, "Enter username and password and login", agent=agent
    )
    actions.input_field("username", "BillyJane")
    actions.input_field("password", "pas$word")
    actions.submit()
    await actions.async_perform()
