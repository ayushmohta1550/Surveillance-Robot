"""Compatibility entry point used by RUN_SURVEILLANCE.ps1."""

from .app.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
