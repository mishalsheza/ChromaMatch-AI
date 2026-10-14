import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Tests must never call Groq for real. An empty value also stops backend/app.py's
# load_dotenv() from pulling a real key out of backend/.env (it doesn't override).
os.environ["GROQ_API_KEY"] = ""


class _NoLiveGroq:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Live Groq call attempted in tests; patch groq_writer.Groq instead")


@pytest.fixture(autouse=True)
def no_live_llm_calls(monkeypatch):
    """Safety net: any un-mocked Groq client construction fails loudly."""
    import ai.recommendation.groq_writer as groq_writer
    monkeypatch.setattr(groq_writer, "Groq", _NoLiveGroq)
    monkeypatch.setenv("GROQ_API_KEY", "")
