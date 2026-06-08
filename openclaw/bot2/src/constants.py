import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(_HERE, 'sessions')
MEMORY_DIR = os.path.join(_HERE, 'memory')
APPROVALS_FILE = os.path.join(_HERE, 'workspace', 'exec-approvals.json')
