#!/usr/bin/env python3
"""
Entrypoint wrapper to run the Gujarat Open Access + IEX AI Bidding Co-Pilot Demo.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from gujarat_oa_copilot.cli import main

if __name__ == "__main__":
    main()
