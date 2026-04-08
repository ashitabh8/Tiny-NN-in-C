"""
Pytest wrapper for standalone C unit tests.

Compiles and runs test_c_ops.c, test_c_ops_int8.c, and test_c_ops_int16.c.
Auto-skips if gcc is not available.
"""

import pytest
import subprocess
import tempfile
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
C_OPS_DIR = os.path.join(REPO_ROOT, "src", "c_ops")
TEST_DIR = os.path.join(REPO_ROOT, "test")


def _gcc_available():
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


def _compile_and_run(c_file):
    """Compile a C test file and run it. Assert exit code 0."""
    basename = os.path.splitext(os.path.basename(c_file))[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        exe = os.path.join(tmpdir, basename)
        result = subprocess.run(
            ["gcc", "-o", exe, c_file, f"-I{C_OPS_DIR}", "-lm", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Compile failed:\n{result.stderr}"

        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"Test failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"


class TestCOpsFloat:
    def test_run(self):
        c_file = os.path.join(TEST_DIR, "test_c_ops.c")
        if not os.path.exists(c_file):
            pytest.skip("test_c_ops.c not found")
        _compile_and_run(c_file)


class TestCOpsInt8:
    def test_run(self):
        c_file = os.path.join(TEST_DIR, "test_c_ops_int8.c")
        if not os.path.exists(c_file):
            pytest.skip("test_c_ops_int8.c not found")
        _compile_and_run(c_file)


class TestCOpsInt16:
    def test_run(self):
        c_file = os.path.join(TEST_DIR, "test_c_ops_int16.c")
        if not os.path.exists(c_file):
            pytest.skip("test_c_ops_int16.c not found")
        _compile_and_run(c_file)
