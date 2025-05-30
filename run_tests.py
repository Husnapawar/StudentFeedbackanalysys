"""
Script to run all tests for the Student Feedback Analysis project.
"""

import unittest
import sys
import os

def run_tests():
    """
    Discover and run all tests in the tests directory.
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the test loader
    loader = unittest.TestLoader()
    
    # Discover tests in the tests directory
    test_suite = loader.discover(os.path.join(script_dir, 'tests'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return the result
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
