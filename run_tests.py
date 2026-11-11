"""
Test Runner Script
Author: Rohit Singh Negi (Reg No: 12311143)
Project: Hospital Appointment No-Show Prediction

Runs the entire automated test suite:
- Pipeline and data integrity tests
- Model loading, calibration, and inference tests
- REST API endpoint integration tests
"""

import unittest
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING AUTOMATED TEST SUITE")
    print("Author: Rohit Singh Negi (Reg No: 12311143)")
    print("="*60 + "\n")
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print(f"TESTS FAILED: Errors={len(result.errors)}, Failures={len(result.failures)}")
        print("="*60 + "\n")
        sys.exit(1)

