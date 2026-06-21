import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from main import main

# Dummy smoke test
def test_main_runs_without_error(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello" in captured.out