"""Streamlit CLI helpers."""
from __future__ import annotations

import argparse
import sys
from typing import List


def streamlit_app_argv() -> List[str]:
    argv = sys.argv[1:]
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv


def parse_streamlit_project_args(
    default_project: str = "data/survey_projects/my_bed",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=default_project)
    return parser.parse_known_args(streamlit_app_argv())[0]
