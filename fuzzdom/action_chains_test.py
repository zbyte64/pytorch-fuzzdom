import pytest
import os
import logging
from .action_chains import FuzzyActionChains
from .env import open_driver
from .dir_paths import TEST_SITE_PATH

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


@pytest.mark.asyncio
async def test_action_chain():
    assert os.path.exists(os.path.join(TEST_SITE_PATH, "login.html"))
    driver = await open_driver()
    # url = f"file://{TEST_SITE_PATH}/login.html"
    url = "http://testsite/login.html"
    await driver.get(url)
    _url = await driver.get_url()
    assert url == _url, _url
    logger.info(url)
    logger.debug(await driver.get_page_source())
    await driver.wait_for_element(1, "input")
    actions = FuzzyActionChains(driver, "Enter username and password and login")
    actions.input_field("username", "BillyJane")
    actions.input_field("password", "pas$word")
    actions.submit()
    score = await actions.async_perform()
    assert score > 0
