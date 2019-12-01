from setuptools import find_packages, setup

setup(
    name="fuzzdom",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        #"pytorch-geometric",
        "arsenic",
        #"pytorch-a2c-ppo-acktr-gail @ git+https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.git@master#egg=pytorch-a2c-ppo-acktr-gail",
        #"miniwob @ git+https://github.com/zbyte64/miniwob-plusplus.git@tweak-fields#egg=miniwob",
    ]
)
