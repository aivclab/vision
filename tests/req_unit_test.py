"""Test availability of required packages."""

import pytest
import unittest
from importlib.metadata import Distribution
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from warg import get_requirements_from_file

_REQUIREMENTS_PATH = Path(__file__).parent.with_name("requirements.txt")
_EXTRA_REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements"


class TestRequirements(unittest.TestCase):
    """Test availability of required packages."""

    def test_requirements(self):
        """Test that each required package is available."""
        requirements = get_requirements_from_file(_REQUIREMENTS_PATH)
        for requirement in requirements:
            with self.subTest(requirement=requirement):
                try:
                    Distribution.from_name(requirement.name)
                except PackageNotFoundError:
                    assert False, f"{requirement} not satisfied"

    @pytest.mark.skipif(True, reason="some platforms might not work")
    @pytest.mark.xfail(
        strict=False
    )  # Does not successfully parse recursions of reqs using -r
    def test_extra_requirements(self):
        """Test that each required package is available."""
        if _EXTRA_REQUIREMENTS_PATH.exists():
            for extra_req_file in _EXTRA_REQUIREMENTS_PATH.iterdir():
                if extra_req_file.is_file() and extra_req_file.suffix == ".txt":
                    requirements = get_requirements_from_file(extra_req_file)
                    for requirement in requirements:
                        with self.subTest(requirement=requirement):
                            try:
                                Distribution.from_name(requirement.name)
                            except PackageNotFoundError:
                                assert False, f"{requirement} not satisfied"
