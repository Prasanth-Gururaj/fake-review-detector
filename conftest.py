# conftest.py  (place in project ROOT, not inside tests/)
import sys
import os

# Add project root to Python path so all modules are findable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
