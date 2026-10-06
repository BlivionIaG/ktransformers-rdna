"""AMD/ROCm source contracts for the phase-0 build.

Runtime GPU tests still need hardware. This file checks the ROCm arch split
and the CPU expert paths without a device.
"""

import os
import sys
import unittest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_amd_ci

# Register this test for AMD CI (estimated time: 10 seconds, placeholder)
# Update suite name when implementing: currently using "stage-a-test-1"
register_amd_ci(est_time=10, suite="stage-a-test-1")


def test_amd_placeholder():
    """ROCm phase-0 source contracts. No GPU required.

    The historical name is kept so the AMD CI registry still collects this file.
    """
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "test_rocm_rdna_phase0.py")
    spec = importlib.util.spec_from_file_location("test_rocm_rdna_phase0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("ROCm phase-0 contract tests failed")


if __name__ == "__main__":
    test_amd_placeholder()
    print("ROCm phase-0 contracts passed")
