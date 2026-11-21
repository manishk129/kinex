"""Environment loading for the Family OS backend.

`server.py` runs with `server/src` on `sys.path`, while tooling may import the
package as `server.src.*`. Keep this module import-safe in both modes.
"""
import os

from dotenv import load_dotenv


def load_env() -> None:
    """Load `server/.env` into the process environment.

    The Agora CLI writes this file, and `server.py` calls this before importing
    the agent. `override=True` keeps local development predictable.
    """
    server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(server_root, ".env"), override=True)
