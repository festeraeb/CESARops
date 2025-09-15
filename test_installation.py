#!/usr/bin/env python3
"""
Installation Test Script
Tests all components and reports status
"""
import sys

def main():
    print(f'Python: {sys.version}')
    
    # Test core dependencies
    try:
        import pandas, numpy, requests, yaml, scipy
        print('✓ Core dependencies OK')
    except ImportError as e:
        print(f'✗ Core dependency error: {e}')
        sys.exit(1)

    # Test ML support
    try:
        import sklearn, joblib
        print('✓ ML support OK')
    except ImportError:
        print('- ML support not available')

    # Test KML support
    try:
        import simplekml
        print('✓ KML support OK')
    except ImportError:
        print('- KML support not available')

    # Test GUI support
    try:
        import tkinter
        print('✓ GUI support OK')
    except ImportError:
        print('✗ GUI support not available')
        sys.exit(1)

    print('Installation test passed!')

if __name__ == '__main__':
    main()
