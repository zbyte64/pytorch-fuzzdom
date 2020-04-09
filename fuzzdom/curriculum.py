import os
from collections import defaultdict
import random

from .dir_paths import MINIWOB_HTML


class LevelTracker:
    global_scoreboard = defaultdict(lambda: {"pass": 0, "fail": 0})

    def __init__(self, levels:list):
        self.levels = levels
        self.current_level_idx = 0

    def select_level(self):
        pass

    @property
    def current_level(self):
        return self.levels[self.current_level_idx]

    def get_level(self):
        return self.current_level

    def __call__(self, pass_or_fail, task_fields=None):
        stats = self.global_scoreboard[self.current_level]
        if pass_or_fail:
            stats["pass"] += 1
            self.current_level_idx += 1
            if self.current_level_idx == len(self.levels):
                self.current_level_idx = random.randint(0, len(self.levels)-1)
        else:
            stats["fail"] += 1
            self.current_level_idx -= 1
        self.current_level_idx = max(0, min(len(self.levels)-1, self.current_level_idx))


def _miniwob_path(f):
    if isinstance(f, list):
        return list(map(_miniwob_path, f))
    p = os.path.join("miniwob", f) + ".html"
    assert os.path.exists(os.path.join(MINIWOB_HTML, p)), p
    return p


MINIWOB_CHALLENGES = _miniwob_path([
    ["click-link", "navigate-tree"],
    ["click-dialog-2", "click-widget"],
    ["click-option"],
    ["click-checkboxes"],# "click-checkboxes-large"],
    ["choose-list"],
    ["enter-text", "enter-text-dynamic", "enter-password"],
    ["login-user", "login-user-popup"],
    #["social-media", "social-media-all", "social-media-some"],
    ["click-tab-2", "click-tab-2-medium"],# "click-tab-2-hard"],
    #["email-inbox-delete", "email-inbox-forward", "email-inbox-important", "email-inbox-noscroll", "email-inbox-reply", "email-inbox-star-reply", "email-inbox"],
    ["search-engine"],
    ["use-autocomplete"],
    #["choose-date-easy", "choose-date-medium", "choose-date"],
    ["multi-layouts", "multi-orderings"],
    ["read-table", "read-table-2"],
    #["book-flight-nodelay"],
    #TODO requires hover action:
    #["click-menu", "click-menu-2"],
])

MINIWOB_TASKS = _miniwob_path([
    ## identify action
    # "click-test",
    # "focus-text",
    ## identify which of N and action
    # only visual differences:
    #"click-test-2",
    # TODO: sometimes broken, generates duplicate button labels
    # "click-button",
    "click-widget",
    "click-link",
    # different meaning of target
    # "focus-text-2",
    # "click-dialog",
    #"click-tab",
    # chrome only
    "click-dialog-2",
    # tokenize color? "click-color",
    #"click-button-sequence",
    ## sequences
    ## sequence and final action
    "click-option",
    "click-checkboxes",
    #"choose-list",
    "enter-text",
    # requires changing case
    # "enter-text-2",
    "enter-text-dynamic",
    "login-user",
    "enter-password",
    #"login-user-popup",
    #"click-checkboxes-large",
    # objectives not parsed:
    # "enter-time",
    ## search for and action
    "navigate-tree",
    "social-media",
    #"click-tab-2-easy",
    #"click-tab-2-medium",
    "click-tab-2",
    "email-inbox-delete",
    #"email-inbox-forward",
    #"email-inbox-important",
    #"email-inbox-noscroll",
    #"email-inbox-star-reply",
    # paste data
    #"multi-layouts",
    #"multi-orderings",
    # incomplete fields
    # "copy-paste",
    # "copy-paste-2",
    #"read-table",
    #"read-table-2",
    #"social-media-all",
    # requires counting onto next page
    "search-engine",
    #"click-tab-2-hard",
    ## custom widget inputs
    #"enter-date",
    ## abstract reasoning, probably impossible
    # objective not parsed: "use-spinner"
    #"email-inbox-reply",
    #"email-inbox",
    #"book-flight-nodelay",
    #"choose-date-nodelay",
    #"click-collapsible-nodelay",
    #"click-collapsible-2-nodelay",
    #"use-autocomplete-nodelay",
    #"click-pie-nodelay",
    # tasks with delays
    #"book-flight",
    #"choose-date-easy",
    #"choose-date",
    #"click-collapsible",
    #"click-collapsible-2",
    "use-autocomplete",
    # "guess-number",
    # "identify-shape",
    # "click-shades",
    # "click-shape",
    # "count-shape",
    # "grid-coordinate",
    # "tic-tac-toe",
])
