"""Per-task turn budget (``set-budget``) — DB layer, worker spawn, CLI.

Covers the t_8041b8cf feature: a per-task ``max_turns`` override that lets one
deep card get more agent iterations than the profile-global ``agent.max_turns``
without inflating every other worker. Mirrors the model-override feature:
kanban_db.set_max_turns(), create_task(max_turns=...), the dispatcher passing
``--max-turns N`` to the worker, and the ``hermes kanban set-budget`` CLI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------


def test_create_task_with_max_turns(conn):
    tid = kb.create_task(conn, title="deep card", assignee="elias", max_turns=150)
    assert kb.get_task(conn, tid).max_turns == 150


def test_set_and_clear_max_turns(conn):
    tid = kb.create_task(conn, title="t", assignee="elias")
    assert kb.get_task(conn, tid).max_turns is None  # default = profile default

    assert kb.set_max_turns(conn, tid, 300)
    assert kb.get_task(conn, tid).max_turns == 300

    # Clearing reverts to the profile default (None).
    assert kb.set_max_turns(conn, tid, None)
    assert kb.get_task(conn, tid).max_turns is None


def test_set_max_turns_non_positive_clears(conn):
    tid = kb.create_task(conn, title="t", assignee="elias", max_turns=150)
    assert kb.set_max_turns(conn, tid, 0)
    assert kb.get_task(conn, tid).max_turns is None
    assert kb.set_max_turns(conn, tid, -5)
    assert kb.get_task(conn, tid).max_turns is None


def test_set_max_turns_unknown_task_returns_false(conn):
    assert kb.set_max_turns(conn, "t_missing", 150) is False


def test_set_max_turns_archived_rejected(conn):
    tid = kb.create_task(conn, title="t", assignee="elias")
    kb.archive_task(conn, tid)
    with pytest.raises(RuntimeError):
        kb.set_max_turns(conn, tid, 150)


def test_migration_adds_max_turns_column(tmp_path, monkeypatch):
    """A DB created before the column existed gets it added on open."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as c:
        c.execute("ALTER TABLE tasks DROP COLUMN max_turns")
        c.commit()
    kb.init_db()  # re-open -> migration path
    with kb.connect() as c:
        cols = {row["name"] for row in c.execute("PRAGMA table_info(tasks)")}
    assert "max_turns" in cols


# ---------------------------------------------------------------------------
# Worker spawn — argv carries --max-turns
# ---------------------------------------------------------------------------


def _spawn_and_capture(monkeypatch, tmp_path, task):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


def test_spawn_passes_max_turns(monkeypatch, tmp_path, conn):
    tid = kb.create_task(conn, title="deep", assignee="elias", max_turns=150)
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    i = cmd.index("--max-turns")
    assert i > cmd.index("chat")  # a chat-subcommand flag, not a global one
    assert cmd[i + 1] == "150"


def test_spawn_omits_max_turns_when_unset(monkeypatch, tmp_path, conn):
    tid = kb.create_task(conn, title="plain", assignee="elias")
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    assert "--max-turns" not in cmd  # other cards keep the profile default


# ---------------------------------------------------------------------------
# CLI — set-budget
# ---------------------------------------------------------------------------


def test_cli_set_budget_and_clear(conn):
    tid = kb.create_task(conn, title="deep", assignee="elias")
    out = kc.run_slash(f"set-budget {tid} 150")
    assert f"Set turn budget on {tid}: 150" in out
    assert kb.get_task(conn, tid).max_turns == 150

    out = kc.run_slash(f"set-budget {tid} none")
    assert "Cleared turn budget" in out
    assert kb.get_task(conn, tid).max_turns is None
