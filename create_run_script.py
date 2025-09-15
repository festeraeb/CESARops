#!/usr/bin/env python3
"""
Create Run Script
Creates the run_cesarops.bat file
"""

def main():
    batch_content = '''@echo off
cd /d "%~dp0"
call venv\\Scripts\\activate.bat
python cesarops_enhanced.py
pause
'''
    
    with open('run_cesarops.bat', 'w') as f:
        f.write(batch_content)
    
    print('Created run_cesarops.bat')

if __name__ == '__main__':
    main()
