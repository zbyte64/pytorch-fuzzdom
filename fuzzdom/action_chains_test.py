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
    actions = FuzzyActionChains(
        driver, "Enter username and password and login", error_value=-5
    )
    actions.input_field("email", "BillyJane")
    actions.input_field("password", "pas$word")
    actions.submit()
    score = await actions.async_perform()
    entered_username = await driver.execute_script(
        "return document.getElementById('email').value;"
    )
    assert entered_username, str(entered_username)
    html = await driver.execute_script("return document.documentElement.innerHTML;")
    assert score > 0, html
