"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


_CFG = {"memory": {"provider": "fakeprovider"}, "agent": {}}


def _make_agent(provider, platform=None, **kwargs):
    with (
        patch("hermes_cli.config.load_config", return_value=_CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=_CFG),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform=platform,
            **kwargs,
        )


class RecordingPreAdmitProvider:
    name = "recording-pre-admit"

    def __init__(self, *, allow=True):
        self._allow = allow
        self.pre_admit_calls = []
        self.is_available_called = False
        self.initialize_called = False

    def pre_admit(self, platform, agent_context):
        self.pre_admit_calls.append((platform, agent_context))
        return self._allow

    def is_available(self):
        self.is_available_called = True
        return True

    def initialize(self, session_id, **kwargs):
        self.initialize_called = True

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_pre_admit_provider_none_maps_to_cli():
    """platform=None is normalised to 'cli' before pre_admit."""
    from agent.agent_init import _pre_admit_provider

    provider = RecordingPreAdmitProvider(allow=True)
    assert _pre_admit_provider(provider, None) is True
    assert provider.pre_admit_calls == [("cli", "primary")]


@pytest.mark.parametrize("falsey_platform", ["", False, 0, [], {}])
def test_pre_admit_provider_falsey_non_none_preserved(falsey_platform):
    """Falsey but non-None platform values reach pre_admit unchanged."""
    from agent.agent_init import _pre_admit_provider

    provider = RecordingPreAdmitProvider(allow=True)
    _pre_admit_provider(provider, falsey_platform)
    assert provider.pre_admit_calls == [(falsey_platform, "primary")]


@pytest.mark.parametrize("set_non_callable", [False, True], ids=["absent", "non-callable"])
def test_pre_admit_provider_absent_or_non_callable_allows(set_non_callable):
    """Absent or non-callable pre_admit defaults to allow."""
    from agent.agent_init import _pre_admit_provider

    duck = SimpleNamespace(name="duck")
    if set_non_callable:
        duck.pre_admit = "not-callable"
    assert _pre_admit_provider(duck, None) is True


# ---------------------------------------------------------------------------
# New regressions required by adversarial review t_174e1eda
# ---------------------------------------------------------------------------


class FalseyBoolProvider:
    """Provider with __bool__=False; tracks lifecycle events via injected list."""

    name = "falsey-bool"

    def __init__(self, events: list):
        self._events = events

    def pre_admit(self, platform, agent_context):
        self._events.append("pre_admit")
        return True

    def is_available(self):
        self._events.append("is_available")
        return True

    def initialize(self, session_id, **kwargs):
        self._events.append("initialize")

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass

    def __bool__(self):
        return False


class FalseyLenProvider:
    """Provider with __len__=0; tracks lifecycle events via injected list."""

    name = "falsey-len"

    def __init__(self, events: list):
        self._events = events

    def pre_admit(self, platform, agent_context):
        self._events.append("pre_admit")
        return True

    def is_available(self):
        self._events.append("is_available")
        return True

    def initialize(self, session_id, **kwargs):
        self._events.append("initialize")

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass

    def __len__(self):
        return 0


@pytest.mark.parametrize(
    "ProviderCls,description",
    [
        (FalseyBoolProvider, "__bool__=False"),
        (FalseyLenProvider, "__len__=0"),
    ],
)
def test_falsey_provider_exact_event_sequence(ProviderCls, description):
    """Falsey non-None provider produces exact event sequence load->pre_admit->is_available->initialize."""
    events: list = []
    provider = ProviderCls(events=events)
    assert not bool(provider), f"provider must be falsey for this test ({description})"

    def _loading_factory(_name):
        events.append("load")
        return provider

    with (
        patch("hermes_cli.config.load_config", return_value=_CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=_CFG),
        patch("plugins.memory.load_memory_provider", side_effect=_loading_factory),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="cli",
        )

    assert events == ["load", "pre_admit", "is_available", "initialize"], (
        f"Expected exact order load->pre_admit->is_available->initialize ({description}), got {events}"
    )


def test_aiagent_exact_event_order_load_pre_admit_is_available_initialize():
    """AIAgent init fires load -> pre_admit -> is_available -> initialize in that exact order."""
    events: list = []

    class OrderProbe:
        name = "order-probe"

        def pre_admit(self, platform, agent_context):
            events.append("pre_admit")
            return True

        def is_available(self):
            events.append("is_available")
            return True

        def initialize(self, session_id, **kwargs):
            events.append("initialize")

        def get_tool_schemas(self):
            return []

        def shutdown(self):
            pass

    probe = OrderProbe()

    def _loading_factory(_name):
        events.append("load")
        return probe

    with (
        patch("hermes_cli.config.load_config", return_value=_CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=_CFG),
        patch("plugins.memory.load_memory_provider", side_effect=_loading_factory),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="cli",
        )

    assert events == ["load", "pre_admit", "is_available", "initialize"], (
        f"Expected exact order load->pre_admit->is_available->initialize, got {events}"
    )


def test_denial_real_path_exact_events_and_no_warning():
    """Denial real-path: exact events [load, pre_admit] only; no is_available/init/unavailable warning."""
    events: list = []
    sentinel_exc = RuntimeError("sentinel-denial")

    class DenyingProvider:
        name = "denying"

        def pre_admit(self, platform, agent_context):
            events.append("pre_admit")
            return False

        def is_available(self):
            events.append("is_available")
            return True

        def initialize(self, session_id, **kwargs):
            events.append("initialize")

        def get_tool_schemas(self):
            return []

        def shutdown(self):
            pass

    def _loading_factory(_name):
        events.append("load")
        return DenyingProvider()

    with (
        patch("hermes_cli.config.load_config", return_value=_CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=_CFG),
        patch("plugins.memory.load_memory_provider", side_effect=_loading_factory),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent.logger") as mock_logger,
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="cli",
        )

    assert events == ["load", "pre_admit"], (
        f"Denial must produce exactly [load, pre_admit]; got {events}"
    )
    assert agent._memory_manager is None or not agent._memory_manager.providers, (
        "manager must be empty after denial"
    )
    # No unavailable warning — denial is a deliberate veto, not a config problem
    for call in mock_logger.warning.call_args_list:
        args = call.args
        assert "unavailable" not in (args[0] if args else "").lower(), (
            f"No unavailable warning expected after denial; got: {args}"
        )


def test_raising_pre_admit_follows_exception_path_no_crash():
    """Raising pre_admit triggers existing exception handler with exact format and sentinel object.

    Asserts:
    - events list is exactly [load, pre_admit] (is_available and initialize not reached)
    - manager is cleared to None
    - warning called once with format 'Memory provider plugin init failed: %s' and the actual exception
    """
    SENTINEL_EXC = RuntimeError("veto via exception")
    events: list = []

    class RaisingProvider:
        name = "raising"

        def pre_admit(self, platform, agent_context):
            events.append("pre_admit")
            raise SENTINEL_EXC

        def is_available(self):
            events.append("is_available")
            return True

        def initialize(self, session_id, **kwargs):
            events.append("initialize")

        def get_tool_schemas(self):
            return []

        def shutdown(self):
            pass

    def _loading_factory(_name):
        events.append("load")
        return RaisingProvider()

    with (
        patch("hermes_cli.config.load_config", return_value=_CFG),
        patch("hermes_cli.config.load_config_readonly", return_value=_CFG),
        patch("plugins.memory.load_memory_provider", side_effect=_loading_factory),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent.logger") as mock_logger,
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform="cli",
        )

    assert events == ["load", "pre_admit"], (
        f"Exact events stop at pre_admit; is_available/initialize must not be called. Got: {events}"
    )
    assert agent._memory_manager is None, "manager must be cleared on exception"
    mock_logger.warning.assert_called_once_with(
        "Memory provider plugin init failed: %s", SENTINEL_EXC
    )


# ---------------------------------------------------------------------------
# Original tests (unchanged)
# ---------------------------------------------------------------------------


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names
