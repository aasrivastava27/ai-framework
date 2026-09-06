import subprocess
import sys


def main():

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui.py",
        ]
    )


if __name__ == "__main__":
    main()