#!/usr/bin/env python3
"""
Tkinter Availability Check
Tests if GUI support is available
"""
import sys

def main():
    try:
        import tkinter
        sys.exit(0)  # Success
    except ImportError:
        sys.exit(1)  # Failure

if __name__ == '__main__':
    main()
