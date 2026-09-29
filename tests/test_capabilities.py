from typing import get_args

from llmserve.client.capabilities import CAPABILITIES, SERVER_KINDS, capabilities
from llmserve.config.schema import Server


def test_every_declared_kind_has_a_capability_entry() -> None:
    assert set(SERVER_KINDS) == set(CAPABILITIES)


def test_config_schema_kinds_match_the_capability_table() -> None:
    # `Server.kind` and SERVER_KINDS are two declarations of one fact; a new backend must
    # update both (they are checked here, not by convention).
    assert get_args(Server.model_fields["kind"].annotation) == SERVER_KINDS


def test_unknown_kind_is_rejected() -> None:
    try:
        capabilities("triton")
    except ValueError as e:
        assert "unknown server kind" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown kind")


def test_gpu_metrics_only_where_a_local_gpu_sampler_makes_sense() -> None:
    gpu = [kind for kind, caps in CAPABILITIES.items() if caps.gpu_metrics]
    assert gpu == ["nim", "vllm"]


def test_prefix_cache_risk_is_limited_to_backends_that_can_have_one() -> None:
    risky = [kind for kind, caps in CAPABILITIES.items() if caps.prefix_cache_risk]
    assert risky == ["nim", "vllm"]


def test_uncontrollable_prefix_cache_is_a_dev_surface_only() -> None:
    uncontrollable = {
        kind
        for kind, caps in CAPABILITIES.items()
        if caps.prefix_cache_risk and not caps.prefix_cache_control
    }
    assert uncontrollable == {"nim"}  # hosted NIM can't turn caching off (ADR-0010)
    assert CAPABILITIES["vllm"].prefix_cache_control


def test_only_mock_and_vllm_advertise_a_health_endpoint() -> None:
    healthy = {kind for kind, caps in CAPABILITIES.items() if caps.health_endpoint}
    assert healthy == {"mock", "vllm"}


def test_token_id_backends_agree_on_forced_output_length() -> None:
    for kind, caps in CAPABILITIES.items():
        if caps.token_id_prompts:
            # ADR-005 pairs exact prompt lengths with exact output lengths.
            assert caps.forced_output_length, kind


def test_mock_and_vllm_share_the_harness_contract() -> None:
    mock, vllm = CAPABILITIES["mock"], CAPABILITIES["vllm"]
    # The client must take the same code path against either backend (ADR-005/ADR-009).
    for field in ("token_id_prompts", "forced_output_length", "usage_block", "model_lookup"):
        assert getattr(mock, field) == getattr(vllm, field), field
