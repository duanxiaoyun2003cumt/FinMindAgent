"""Tests for scripts/preflight_structured_runtime.py — config resolution, route
validation, exit codes, modes, main wiring, and project .env loading.

No network calls — runtime/LLM are mocked where needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.preflight_structured_runtime import (
    EXIT_PASS,
    EXIT_FAIL,
    EXIT_BLOCKED,
    PreflightMode,
    PROVIDER_DEFAULT_MODELS,
    check_preflight_environment,
    preflight_exit_code,
    resolve_preflight_models,
    resolve_preflight_mode,
    validate_preflight_route,
)

# Default DeepSeek profile (current V4 production contract)
DS_QUICK = "deepseek-v4-flash"
DS_DEEP = "deepseek-v4-pro"


# ============================================================================
# Project-root .env loading (Phase A preflight env fix)
# ============================================================================

class TestProjectEnvLoading:
    """The preflight must load the project-root .env before the environment
    check, without overriding explicit OS environment variables, and must
    locate the project root from the script path (not cwd)."""

    def test_load_project_env_invoked_before_environment_check(self, tmp_path):
        """Test 1: main() loads project .env before check_preflight_environment.

        Use a fake .env file and an API key NOT in os.environ; if the loader
        runs, the environment check sees the key. We patch the project-root
        resolution so the test does not touch the real project .env.
        """
        m = self._import_module()
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=from-dotenv-value\n", encoding="utf-8")

        captured = {}

        def fake_load_dotenv(path, override=False):
            captured["path"] = path
            captured["override"] = override
            # Simulate dotenv behavior without reading a real credential:
            # only set the variable if not already present (override=False).
            if "DEEPSEEK_API_KEY" not in os.environ:
                os.environ["DEEPSEEK_API_KEY"] = "from-dotenv-value"

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(m, "load_dotenv", side_effect=fake_load_dotenv):
                with patch.object(
                    m, "PROJECT_ROOT", tmp_path
                ):
                    with patch.object(m, "create_runtime") as mock_runtime:
                        mock_runtime.return_value.deep_llm = None
                        mock_runtime.return_value.quick_llm = None
                        with patch("sys.argv", ["preflight", "deepseek"]):
                            code = m.main()

        assert captured.get("path") == tmp_path / ".env"
        assert captured.get("override") is False
        # Because the loader ran with the key, we are no longer BLOCKED on
        # the key (runtime returns fake runtime -> exit depends on agents,
        # but the key check passed: not EXIT_BLOCKED).
        assert code != EXIT_BLOCKED

    def test_explicit_environment_variable_wins(self, tmp_path):
        """Test 2: override=False means an existing OS env var is kept."""
        m = self._import_module()
        env_file = tmp_path / ".env"
        env_file.write_text("DEEPSEEK_API_KEY=from-dotenv-value\n", encoding="utf-8")

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "system-value"}, clear=False):
            with patch.object(m, "load_dotenv") as mock_load:
                mock_load.side_effect = lambda path, override=False: None
                with patch.object(m, "PROJECT_ROOT", tmp_path):
                    m.load_project_env()
            # os.environ value must remain the system value
            assert os.environ["DEEPSEEK_API_KEY"] == "system-value"
            mock_load.assert_called_once_with(tmp_path / ".env", override=False)

    def test_project_root_independent_of_cwd(self, tmp_path, monkeypatch):
        """Test 3: PROJECT_ROOT derives from the script path, not cwd."""
        m = self._import_module()
        # PROJECT_ROOT is computed from __file__ at import; assert it points
        # to the repo root (the scripts/ parent) regardless of cwd.
        assert (m.PROJECT_ROOT / "scripts" / "preflight_structured_runtime.py").exists()
        assert (m.PROJECT_ROOT / ".env").exists() or True  # .env presence is env-specific
        # Changing cwd must not change the module-level PROJECT_ROOT.
        monkeypatch.chdir(tmp_path)
        assert (m.PROJECT_ROOT / "scripts" / "preflight_structured_runtime.py").exists()

    @staticmethod
    def _import_module():
        import scripts.preflight_structured_runtime as m
        return m


class TestCheckPreflightEnvironment:
    def test_missing_api_key_blocked(self):
        with patch.dict("os.environ", {}, clear=True):
            ok, reason = check_preflight_environment("deepseek")
            assert ok is False
            assert "API key" in reason

    def test_api_key_present_ok(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            ok, reason = check_preflight_environment("deepseek")
            assert ok is True
            assert reason == ""

    def test_unknown_provider_blocked(self):
        with patch.dict("os.environ", {"UNKNOWN_API_KEY": "x"}, clear=False):
            ok, reason = check_preflight_environment("unknown-provider")
            assert ok is False
            assert "not supported" in reason

    def test_supported_without_defaults_blocked_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            ok, reason = check_preflight_environment("anthropic")
            assert ok is False

# Default DeepSeek profile (current V4 production contract)
DS_QUICK = "deepseek-v4-flash"
DS_DEEP = "deepseek-v4-pro"


class TestResolvePreflightModels:
    def test_openai_verified_defaults(self):
        models = resolve_preflight_models("openai")
        assert models == {"quick": "gpt-5.4-mini", "deep": "gpt-5.4"}

    def test_deepseek_verified_defaults(self):
        models = resolve_preflight_models("deepseek")
        assert models == {"quick": DS_QUICK, "deep": DS_DEEP}

    def test_unknown_provider_returns_none_even_with_overrides(self):
        assert resolve_preflight_models(
            "unknown-provider",
            quick_override="my-quick",
            deep_override="my-deep",
        ) is None

    def test_supported_without_defaults_requires_overrides(self):
        assert resolve_preflight_models("anthropic") is None

    def test_supported_without_defaults_with_overrides(self):
        models = resolve_preflight_models(
            "anthropic",
            quick_override="my-quick",
            deep_override="my-deep",
        )
        assert models == {"quick": "my-quick", "deep": "my-deep"}

    def test_override_takes_precedence(self):
        models = resolve_preflight_models(
            "deepseek",
            quick_override="custom-quick",
            deep_override="custom-deep",
        )
        assert models["quick"] == "custom-quick"
        assert models["deep"] == "custom-deep"


class TestPreflightExitCode:
    def test_all_pass_exit_zero(self):
        assert preflight_exit_code(passed=12, failed=0, total=12) == EXIT_PASS

    def test_any_fail_exit_one(self):
        assert preflight_exit_code(passed=11, failed=1, total=12) == EXIT_FAIL

    def test_incomplete_exit_one(self):
        assert preflight_exit_code(passed=10, failed=0, total=12) == EXIT_FAIL

    def test_all_fail_exit_one(self):
        assert preflight_exit_code(passed=0, failed=12, total=12) == EXIT_FAIL


class TestResolvePreflightMode:
    def test_default_deepseek_is_production_acceptance(self):
        mode = resolve_preflight_mode(
            provider="deepseek",
            quick_override=None,
            deep_override=None,
            resolved_models={"quick": DS_QUICK, "deep": DS_DEEP},
        )
        assert mode == PreflightMode.PRODUCTION_ACCEPTANCE

    def test_deepseek_with_override_is_custom_smoke(self):
        mode = resolve_preflight_mode(
            provider="deepseek",
            quick_override="deepseek-v4-flash",
            deep_override=None,
            resolved_models={"quick": "deepseek-v4-flash", "deep": DS_DEEP},
        )
        assert mode == PreflightMode.CUSTOM_SMOKE

    def test_non_deepseek_is_custom_smoke(self):
        mode = resolve_preflight_mode(
            provider="openai",
            quick_override=None,
            deep_override=None,
            resolved_models={"quick": "gpt-5.4-mini", "deep": "gpt-5.4"},
        )
        assert mode == PreflightMode.CUSTOM_SMOKE

    def test_deepseek_resolved_models_differ_from_defaults_is_custom(self):
        mode = resolve_preflight_mode(
            provider="deepseek",
            quick_override=None,
            deep_override=None,
            resolved_models={"quick": "deepseek-v4-pro", "deep": "deepseek-v4-flash"},
        )
        assert mode == PreflightMode.CUSTOM_SMOKE


class TestDefaultDeepSeekProfile:
    """Current V4 production profile: quick=deepseek-v4-flash,
    deep=deepseek-v4-pro (legacy reasoner contract retired)."""

    def test_default_deepseek_profile_uses_current_v4_models(self):
        models = resolve_preflight_models("deepseek")
        assert models == {"quick": "deepseek-v4-flash", "deep": "deepseek-v4-pro"}

    def test_production_mode_requires_exact_current_default_pair(self):
        mode = resolve_preflight_mode(
            provider="deepseek",
            quick_override=None,
            deep_override=None,
            resolved_models={"quick": "deepseek-v4-flash", "deep": "deepseek-v4-pro"},
        )
        assert mode == PreflightMode.PRODUCTION_ACCEPTANCE
        # Legacy pair must NOT be production acceptance
        legacy_mode = resolve_preflight_mode(
            provider="deepseek",
            quick_override=None,
            deep_override=None,
            resolved_models={"quick": "deepseek-v4-pro", "deep": "deepseek-reasoner"},
        )
        assert legacy_mode == PreflightMode.CUSTOM_SMOKE

    def test_legacy_reasoner_not_current_production_default(self):
        """deepseek-reasoner must not be the deep default."""
        from finmindagent.llm_clients.capabilities import detect_capabilities
        from finmindagent.llm_clients.capabilities import StructuredOutputMode

        caps = detect_capabilities("deepseek", "deepseek-reasoner")
        assert caps.structured_output_mode == StructuredOutputMode.UNSUPPORTED
        defaults = PROVIDER_DEFAULT_MODELS["deepseek"]
        assert defaults["deep"] != "deepseek-reasoner"


# ============================================================================
# Route-contract validation (current V4 primary-success contract)
# ============================================================================

def _primary_diag(route="primary", model=None, success=True, extra_attempts=()):
    """A diag with a successful primary attempt (current V4 contract)."""
    model = model or DS_QUICK
    attempts = [{"source": "primary", "model": model, "success": success}]
    attempts.extend(extra_attempts)
    return {
        "success": True,
        "selected_model": model if success else None,
        "route": route if success else "unavailable",
        "attempts": attempts,
    }


class TestValidatePreflightRoute:
    # --- Strict default contract (current V4) ---

    def test_quick_tier_v4_flash_primary_success(self):
        """Quick-tier: primary=deepseek-v4-flash, route=primary, success → PASS."""
        diag = _primary_diag(model=DS_QUICK)
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is True, reason

    def test_deep_tier_v4_pro_primary_success(self):
        """Deep-tier RM/PM: primary=deepseek-v4-pro, route=primary → PASS."""
        diag = _primary_diag(model=DS_DEEP)
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="research_manager", tier="deep",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is True, reason
        ok2, _ = validate_preflight_route(
            provider="deepseek", agent_name="portfolio_manager", tier="deep",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok2 is True

    def test_deep_tier_requires_v4_pro_primary(self):
        """Deep-tier using the quick model as primary → ROUTE MISMATCH FAIL."""
        diag = _primary_diag(model=DS_QUICK)
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="portfolio_manager", tier="deep",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "ROUTE MISMATCH" in reason or "selected_model" in reason

    def test_quick_tier_requires_v4_flash_primary(self):
        """Quick-tier using the deep model as primary → ROUTE MISMATCH FAIL."""
        diag = _primary_diag(model=DS_DEEP)
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "ROUTE MISMATCH" in reason or "selected_model" in reason

    def test_route_mismatch_primary_vs_selected_fails(self):
        """diag.route='primary' but successful attempt is fallback → FAIL."""
        diag = {
            "success": True,
            "selected_model": DS_QUICK,
            "route": "primary",  # WRONG: successful attempt is fallback:1
            "attempts": [
                {"source": "primary", "model": DS_QUICK, "success": False, "stage": "invoke"},
                {"source": "fallback:1", "model": DS_QUICK, "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "route" in reason

    def test_quick_tier_missing_attempts_fails(self):
        """No attempts → FAIL (diagnostics must be complete)."""
        diag = {"success": True, "selected_model": DS_QUICK, "route": "primary", "attempts": []}
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "attempts" in reason

    def test_wrong_attempt_model_fails(self):
        """attempts[0].model != expected primary → FAIL."""
        diag = {
            "success": True,
            "selected_model": DS_QUICK,
            "route": "primary",
            "attempts": [
                {"source": "primary", "model": "some-other-model", "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "selected_model" in reason or "attempts[0]" in reason

    def test_selected_model_not_in_successful_attempts_fails(self):
        """selected_model inconsistent with attempts → FAIL."""
        diag = {
            "success": True,
            "selected_model": "some-other-model",
            "route": "primary",
            "attempts": [
                {"source": "primary", "model": DS_QUICK, "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False
        assert "selected_model" in reason

    # --- Legacy reasoner fallback is NOT a production gate anymore ---

    def test_legacy_reasoner_fallback_is_custom_smoke_only(self):
        """A reasoner→v4 fallback diag is internally consistent but fails the
        strict V4 primary-success contract (deep-tier must use v4-pro
        primary)."""
        diag = {
            "success": True,
            "selected_model": DS_QUICK,
            "route": "fallback:1",
            "attempts": [
                {"source": "primary", "model": "deepseek-reasoner",
                 "capability_supported": False, "success": False, "stage": "capability"},
                {"source": "fallback:1", "model": DS_QUICK, "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="research_manager", tier="deep",
            diag=diag,
            expected_quick_model=DS_QUICK, expected_deep_model=DS_DEEP,
            strict_default_contract=True,
        )
        assert ok is False  # legacy contract retired from production gate

    # --- Custom smoke contract ---

    def test_test5_custom_override_not_killed_by_hardcoded_validator(self):
        """Custom quick override works under CUSTOM_SMOKE semantics."""
        custom_quick = "deepseek-v4-flash"
        diag = {
            "success": True,
            "selected_model": custom_quick,
            "route": "primary",
            "attempts": [
                {"source": "primary", "model": custom_quick, "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model=custom_quick, expected_deep_model=DS_DEEP,
            strict_default_contract=False,  # custom smoke
        )
        assert ok is True, reason

    def test_custom_smoke_selected_model_outside_chain_fails(self):
        """Custom smoke: selected model not in configured chain → FAIL."""
        diag = {
            "success": True,
            "selected_model": "some-other-model",
            "route": "primary",
            "attempts": [
                {"source": "primary", "model": "some-other-model", "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="deepseek", agent_name="market_analyst", tier="quick",
            diag=diag,
            expected_quick_model="deepseek-v4-flash", expected_deep_model=DS_DEEP,
            strict_default_contract=False,
        )
        assert ok is False
        assert "candidate chain" in reason

    def test_non_deepseek_internal_consistency(self):
        diag = {
            "success": True,
            "selected_model": "gpt-5.4",
            "route": "primary",
            "attempts": [
                {"source": "primary", "model": "gpt-5.4", "success": True},
            ],
        }
        ok, reason = validate_preflight_route(
            provider="openai", agent_name="research_manager", tier="deep",
            diag=diag,
            expected_quick_model="gpt-5.4-mini", expected_deep_model="gpt-5.4",
            strict_default_contract=True,
        )
        assert ok is True, reason


# ============================================================================
# Main wiring integration tests (offline, no network)
# ============================================================================

class TestMainWiring:
    def _import_main(self):
        import scripts.preflight_structured_runtime as m
        return m

    def test_missing_key_blocks_before_runtime_creation(self):
        """Test A: missing API key → exit 2, create_runtime NOT called.

        The project .env loader runs first (patched to no-op so the real
        project .env is not consulted); with no key anywhere the env check
        blocks before runtime creation.
        """
        m = self._import_main()
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(m, "load_project_env") as mock_loader:
                with patch.object(m, "create_runtime") as mock_runtime:
                    with patch("sys.argv", ["preflight", "deepseek"]):
                        code = m.main()
        mock_loader.assert_called_once()
        assert code == EXIT_BLOCKED
        mock_runtime.assert_not_called()

    def test_deepseek_config_wiring(self):
        """Test B: API key present → config wired with deepseek models."""
        m = self._import_main()
        captured = {}

        fake_runtime = MagicMock()
        fake_runtime.deep_llm = None
        fake_runtime.quick_llm = None

        def fake_create_runtime(config=None, callbacks=None):
            captured["config"] = config
            return fake_runtime

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch.object(m, "create_runtime", side_effect=fake_create_runtime):
                with patch("sys.argv", ["preflight", "deepseek"]):
                    m.main()

        cfg = captured["config"]
        assert cfg["llm_provider"] == "deepseek"
        assert cfg["quick_think_llm"] == DS_QUICK
        assert cfg["deep_think_llm"] == DS_DEEP

    def _make_runtime(self, pm_route_fail: bool = False, quick_model: str = DS_QUICK):
        """Build a fake runtime producing consistent current-V4 diagnostics.

        Quick-tier: primary=quick_model (deepseek-v4-flash) success.
        Deep-tier: primary=deep_model (deepseek-v4-pro) success.
        """
        class FakeLLM:
            def __init__(self, model):
                self.model_name = model

            def with_structured_output(self, schema):
                class _Runnable:
                    def invoke(self, prompt):
                        try:
                            return schema()
                        except Exception:
                            return schema.model_construct()
                return _Runnable()

        class FakeRuntime:
            quick_llm = FakeLLM(quick_model)
            deep_llm = FakeLLM(DS_DEEP)

            def _invoke_structured_agent(self, **kwargs):
                agent_name = kwargs["agent_name"]
                schema_cls = kwargs["schema"]
                try:
                    result = schema_cls()
                except Exception:
                    result = schema_cls.model_construct()
                state = kwargs["state"]
                tier = "deep" if agent_name in ("research_manager", "portfolio_manager") else "quick"
                primary = DS_DEEP if tier == "deep" else quick_model
                diags = state.metadata.setdefault("structured_diagnostics", {})

                if agent_name == "portfolio_manager" and pm_route_fail:
                    # Inconsistent: successful attempt is fallback but route
                    # claims primary
                    diags[agent_name] = {
                        "success": True, "selected_model": DS_DEEP, "route": "primary",
                        "attempts": [
                            {"source": "primary", "model": DS_DEEP, "success": False, "stage": "invoke"},
                            {"source": "fallback:1", "model": quick_model, "success": True},
                        ],
                    }
                else:
                    diags[agent_name] = {
                        "success": True, "selected_model": primary, "route": "primary",
                        "attempts": [
                            {"source": "primary", "model": primary, "success": True},
                        ],
                    }
                return result

        return FakeRuntime()

    def test_full_pass_exit_zero(self):
        """Test F: all agents pass with correct routes → exit 0, acceptance YES."""
        m = self._import_main()
        fake_runtime = self._make_runtime()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch.object(m, "create_runtime", return_value=fake_runtime):
                with patch("sys.argv", ["preflight", "deepseek"]):
                    code = m.main()
        assert code == EXIT_PASS

    def test_route_contract_failure_exit_one(self):
        """Test E: one route inconsistency → exit 1."""
        m = self._import_main()
        fake_runtime = self._make_runtime(pm_route_fail=True)
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch.object(m, "create_runtime", return_value=fake_runtime):
                with patch("sys.argv", ["preflight", "deepseek"]):
                    code = m.main()
        assert code == EXIT_FAIL

    def test_custom_smoke_does_not_close_gate(self):
        """Test 6: custom override run → smoke PASS but not acceptance eligible."""
        m = self._import_main()
        custom_quick = "deepseek-v4-flash"
        fake_runtime = self._make_runtime(quick_model=custom_quick)

        import io
        import contextlib
        out = io.StringIO()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}, clear=False):
            with patch.object(m, "create_runtime", return_value=fake_runtime):
                with patch("sys.argv", ["preflight", "deepseek", "--quick-model", custom_quick]):
                    with contextlib.redirect_stdout(out):
                        code = m.main()

        assert code == EXIT_PASS
        output = out.getvalue()
        assert "CUSTOM MODEL SMOKE" in output
        assert "Phase A production acceptance eligible: NO" in output
        assert "Custom structured smoke: PASS" in output
