"""Isolated fixtures; also usable without pytest via unittest discovery."""
import importlib.util
from pathlib import Path
import os
import tempfile
import sys
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsolatedHome(unittest.TestCase):
    def setUp(self):
        scratch = (Path("/private/tmp/hermes-plugin-rapid-mlx-tests") if sys.platform == "darwin"
                   else Path(tempfile.gettempdir()) / "hermes-plugin-rapid-mlx-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="rapid-mlx-test-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name) / "profile"
        self.env = patch.dict(os.environ, {"HERMES_HOME": str(self.home)})
        self.env.start()
        self.addCleanup(self.env.stop)
