import os, sys
import miniwob


MINIWOB = os.path.dirname(os.path.abspath(miniwob.__file__))
MINIWOB_HTML = os.path.join(MINIWOB, "html")

ROOT_DIR = os.sep + os.path.join(*(os.path.abspath(__file__).split(os.sep)[:-2]))

JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js")
DATA_DIR = os.path.join(ROOT_DIR, "datadir")
APP_PATH = os.path.join(ROOT_DIR, "fuzzdom")
PRETRAINED_PATH = os.path.join(APP_PATH, "pretrained")
