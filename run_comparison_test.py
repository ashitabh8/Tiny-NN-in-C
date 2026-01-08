#!/usr/bin/env python3
"""
Run the PyTorch vs C output comparison test

This is a standalone script to easily run the comparison tests
that verify the C model produces the same output as PyTorch.
"""

import sys
import subprocess

def main():
    print("=" * 70)
    print("Running PyTorch vs C Output Comparison Tests")
    print("=" * 70)
    print()
    print("This test will:")
    print("  1. Compile each model to C")
    print("  2. Run PyTorch inference")
    print("  3. Compile and run the C code")
    print("  4. Compare outputs with error tolerance")
    print()
    print("Requirements: gcc must be installed")
    print("=" * 70)
    print()
    
    # Run the specific comparison tests
    cmd = [
        "pytest",
        "test/test_integration.py::TestPyTorchCComparison",
        "-v",
        "-s"  # Show print statements
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("✓ All comparison tests passed!")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("✗ Some tests failed")
        print("=" * 70)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

