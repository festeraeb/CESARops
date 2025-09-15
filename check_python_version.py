#!/usr/bin/env python3
"""
Python Version Check
Checks if Python version is 3.8 or later
"""
import sys

def main():
    if sys.version_info >= (3, 8):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == '__main__':
    main()
