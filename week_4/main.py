# main.py
"""
Entrypoint helper. Keeps `streamlit run streamlit_app.py` as the canonical way
to run the app, but allows `python main.py` to launch the same thing.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    # Prefer STREAMLIT_RUN env var or default
    script = "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", script]
    subprocess.run(cmd, check=True)
