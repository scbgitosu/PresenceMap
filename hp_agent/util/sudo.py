"""Validate that ``sudo iw`` (and friends) work without prompting.

The agent requires non-interactive sudo for ``iw`` (scanning, monitor mode)
and for reading the ath9k debugfs noise floor. If sudoers isn't configured,
the loop would silently stall on a password prompt.
"""
from __future__ import annotations

import shutil
import subprocess


def has_passwordless_sudo(cmd: str = "iw") -> tuple[bool, str]:
    """Return ``(ok, detail)``. ``ok=True`` means ``sudo -n <cmd> --version`` ran."""
    if shutil.which("sudo") is None:
        return False, "sudo not on PATH"
    if shutil.which(cmd) is None:
        return False, f"{cmd} not on PATH"
    try:
        r = subprocess.run(
            ["sudo", "-n", cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        return False, f"sudo -n {cmd} timed out (likely password prompt)"
    if r.returncode == 0:
        return True, f"{cmd} runs under sudo -n"
    return False, (r.stderr.strip() or r.stdout.strip() or "non-zero exit")
