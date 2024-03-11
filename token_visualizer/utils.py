#!/usr/bin/env python3

import os
from typing import Union

from dotenv import load_dotenv

import token_visualizer

__all__ = ["css_style", "ensure_os_env"]


def css_style() -> str:
    with open(os.path.join(os.path.dirname(token_visualizer.__file__), "main.css")) as f:
        css_style = f.read()
    css_html = f"<style>{css_style}</style>"""
    return css_html


def ensure_os_env(env_name: str, default_value: Union[str, None] = None):
    if env_name in os.environ:
        env_value = os.getenv(env_name)
    else:
        load_dotenv()
        env_value = os.getenv(env_name, default_value)
    return env_value
