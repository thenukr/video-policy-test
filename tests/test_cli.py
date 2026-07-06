import argparse

import pytest

from video_policy_test.cli import _parse_tasks


def test_parse_tasks_deduplicates_and_sorts() -> None:
    assert _parse_tasks("3,1,3") == [1, 3]


def test_parse_tasks_rejects_non_integer() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_tasks("one")

