"""Project paths and fixed surveillance configuration.

All filesystem knowledge lives here so workflow modules do not repeat path
construction or depend on the current working directory.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    workspace: Path
    source: Path
    sapiensid: Path
    openpar_promptpar: Path
    database: Path
    input: Path
    output: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        source = Path(__file__).resolve().parents[1]
        workspace = source.parent
        return cls(
            workspace=workspace,
            source=source,
            sapiensid=workspace / "sapiensid",
            openpar_promptpar=workspace / "OpenPAR" / "PromptPAR",
            database=workspace / "Database",
            input=workspace / "input",
            output=workspace / "output",
        )

    @property
    def sapiens_runner(self) -> Path:
        return self.source / "vision" / "identity.py"

    @property
    def openpar_runner(self) -> Path:
        return self.openpar_promptpar / "test_in_custom.py"

    @property
    def openpar_checkpoint(self) -> Path:
        return self.openpar_promptpar / "PETA_Checkpoint.pth"


PATHS = ProjectPaths.discover()
RECIPIENT = "ayushmohta2402@gmail.com"
TIMEZONE_NAME = "Asia/Kolkata"
