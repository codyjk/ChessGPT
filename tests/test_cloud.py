"""Tests for the cloud training abstraction.

Tests are split into categories:
- Dataclass construction and validation
- Cost estimation (pure functions, no network)
- JSONL merge logic (deduplication by timestamp)
- Provider registry (lazy imports, error handling)
- CLI argument parsing
- SSH helpers (exclusion patterns, mkdir -p)
- Pod state persistence (save/load/clear, InstanceInfo conversion)
- Orchestration integration (mock provider + mock SSH, tmux-based training)
- Cloud lifecycle commands (status, deprovision)

No live provider tests -- those cost money and require API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chessgpt.cloud.pricing import (
    REFERENCE_PRICES,
    TRAINING_TIME_ESTIMATES,
    CostEstimate,
    estimate_cost,
    format_cost_estimate,
    format_cost_summary,
)
from chessgpt.cloud.provider import (
    CloudProvider,
    GpuOffer,
    InstanceInfo,
    InstanceSpec,
    ProviderStatus,
)
from chessgpt.cloud.providers import get_provider, list_providers
from chessgpt.cloud.runner import _detect_config_size, _merge_jsonl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spec() -> InstanceSpec:
    return InstanceSpec(gpu_type="A100", gpu_count=1, disk_gb=50)


@pytest.fixture
def sample_info() -> InstanceInfo:
    return InstanceInfo(
        instance_id="pod-abc123",
        host="ssh.example.com",
        port=22222,
        username="root",
        ssh_key_path="~/.ssh/id_ed25519",
        gpu_type="A100",
        cost_per_hour=1.60,
    )


@pytest.fixture
def sample_status() -> ProviderStatus:
    return ProviderStatus(
        instance_id="pod-abc123",
        state="running",
        uptime_seconds=3600.0,
        estimated_cost=1.60,
    )


@pytest.fixture
def sample_offer() -> GpuOffer:
    return GpuOffer(
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        cost_per_hour=1.60,
        vram_gb=80,
    )


class FakeProvider(CloudProvider):
    """Minimal concrete provider for testing the ABC contract."""

    def __init__(self) -> None:
        self.provisioned: list[InstanceSpec] = []
        self.deprovisioned: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def list_gpu_offers(self, gpu_type: str | None = None) -> list[GpuOffer]:
        return [
            GpuOffer(gpu_type="FAKE_GPU", gpu_count=1, cost_per_hour=0.50, vram_gb=24),
        ]

    def provision(self, spec: InstanceSpec) -> InstanceInfo:
        self.provisioned.append(spec)
        return InstanceInfo(
            instance_id="fake-001",
            host="fake.host",
            port=22,
            username="root",
            ssh_key_path="/tmp/fake_key",
            gpu_type=spec.gpu_type,
            cost_per_hour=0.50,
        )

    def status(self, instance_id: str) -> ProviderStatus:
        return ProviderStatus(instance_id=instance_id, state="running")

    def deprovision(self, instance_id: str) -> None:
        self.deprovisioned.append(instance_id)


# ---------------------------------------------------------------------------
# Dataclass construction tests
# ---------------------------------------------------------------------------


class TestInstanceSpec:
    def test_required_fields(self) -> None:
        spec = InstanceSpec(gpu_type="A100")
        assert spec.gpu_type == "A100"
        assert spec.gpu_count == 1
        assert spec.disk_gb == 50

    def test_custom_fields(self) -> None:
        spec = InstanceSpec(gpu_type="H100", gpu_count=4, disk_gb=200, image="custom:latest")
        assert spec.gpu_count == 4
        assert spec.disk_gb == 200
        assert spec.image == "custom:latest"

    def test_default_image_is_pytorch(self) -> None:
        """Default image should be a PyTorch CUDA image for GPU training."""
        spec = InstanceSpec(gpu_type="A100")
        assert "pytorch" in spec.image
        assert "cuda" in spec.image


class TestInstanceInfo:
    def test_all_fields_stored(self, sample_info: InstanceInfo) -> None:
        assert sample_info.instance_id == "pod-abc123"
        assert sample_info.host == "ssh.example.com"
        assert sample_info.port == 22222
        assert sample_info.username == "root"
        assert sample_info.ssh_key_path == "~/.ssh/id_ed25519"
        assert sample_info.gpu_type == "A100"
        assert sample_info.cost_per_hour == 1.60

    def test_nonstandard_port(self) -> None:
        """RunPod uses non-standard SSH ports via TCP proxy."""
        info = InstanceInfo(
            instance_id="x",
            host="proxy.runpod.net",
            port=43821,
            username="root",
            ssh_key_path="/key",
            gpu_type="A100",
            cost_per_hour=1.0,
        )
        assert info.port == 43821


class TestProviderStatus:
    def test_defaults(self) -> None:
        status = ProviderStatus(instance_id="x", state="provisioning")
        assert status.uptime_seconds == 0.0
        assert status.estimated_cost == 0.0

    def test_running_with_cost(self, sample_status: ProviderStatus) -> None:
        assert sample_status.state == "running"
        assert sample_status.uptime_seconds == 3600.0
        assert sample_status.estimated_cost == 1.60


class TestGpuOffer:
    def test_defaults(self) -> None:
        offer = GpuOffer(gpu_type="A100", gpu_count=1, cost_per_hour=1.60)
        assert offer.vram_gb == 0
        assert offer.available is True
        assert offer.extra == {}

    def test_extra_metadata(self) -> None:
        offer = GpuOffer(
            gpu_type="A100",
            gpu_count=1,
            cost_per_hour=1.60,
            extra={"runpod_id": "NVIDIA A100 80GB PCIe"},
        )
        assert offer.extra["runpod_id"] == "NVIDIA A100 80GB PCIe"


# ---------------------------------------------------------------------------
# CloudProvider ABC tests
# ---------------------------------------------------------------------------


class TestCloudProviderABC:
    def test_concrete_provider_satisfies_abc(self) -> None:
        """FakeProvider implements all abstract methods without error."""
        provider = FakeProvider()
        assert provider.name == "fake"
        offers = provider.list_gpu_offers()
        assert len(offers) == 1

    def test_provision_returns_instance_info(self, sample_spec: InstanceSpec) -> None:
        provider = FakeProvider()
        info = provider.provision(sample_spec)
        assert isinstance(info, InstanceInfo)
        assert info.instance_id == "fake-001"

    def test_deprovision_is_idempotent(self) -> None:
        """Calling deprovision multiple times should not raise."""
        provider = FakeProvider()
        provider.deprovision("fake-001")
        provider.deprovision("fake-001")
        assert provider.deprovisioned == ["fake-001", "fake-001"]

    def test_provision_records_spec(self, sample_spec: InstanceSpec) -> None:
        provider = FakeProvider()
        provider.provision(sample_spec)
        assert len(provider.provisioned) == 1
        assert provider.provisioned[0].gpu_type == "A100"


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_known_gpu_known_config(self) -> None:
        est = estimate_cost("A100", "medium")
        assert est is not None
        assert est.gpu_type == "A100"
        assert est.config_size == "medium"
        assert est.cost_per_hour == REFERENCE_PRICES["A100"]
        assert est.estimated_hours == TRAINING_TIME_ESTIMATES["A100"]["medium"]
        assert est.estimated_cost == est.estimated_hours * est.cost_per_hour

    def test_known_gpu_unknown_config(self) -> None:
        """Unknown config size should still return an estimate with 0 hours."""
        est = estimate_cost("A100", "xlarge")
        assert est is not None
        assert est.estimated_hours == 0.0
        assert est.estimated_cost == 0.0

    def test_unknown_gpu(self) -> None:
        """Unknown GPU type returns None (we have no reference price)."""
        est = estimate_cost("TPU_v5", "medium")
        assert est is None

    def test_cost_estimate_property(self) -> None:
        est = CostEstimate(
            gpu_type="A100",
            config_size="large",
            estimated_hours=7.0,
            cost_per_hour=1.60,
        )
        assert est.estimated_cost == pytest.approx(11.20)

    def test_all_reference_prices_positive(self) -> None:
        """Every reference GPU price should be a positive number."""
        for gpu, price in REFERENCE_PRICES.items():
            assert price > 0, f"{gpu} has non-positive price: {price}"

    def test_training_time_estimates_sorted_by_size(self) -> None:
        """For any given GPU, training time should increase: tiny < small < medium < large."""
        for gpu, times in TRAINING_TIME_ESTIMATES.items():
            sizes = ["tiny", "small", "medium", "large"]
            available = [s for s in sizes if s in times]
            hours = [times[s] for s in available]
            assert hours == sorted(hours), f"{gpu} times are not monotonically increasing"


class TestFormatCostSummary:
    def test_contains_key_fields(self) -> None:
        summary = format_cost_summary("A100", 2.5, 1.60)
        assert "A100" in summary
        assert "2.50" in summary
        assert "1.60" in summary
        assert "$4.00" in summary

    def test_zero_duration(self) -> None:
        summary = format_cost_summary("RTX_4090", 0.0, 0.45)
        assert "$0.00" in summary


class TestFormatCostEstimate:
    def test_contains_config_and_gpu(self) -> None:
        est = CostEstimate(
            gpu_type="H100",
            config_size="large",
            estimated_hours=4.5,
            cost_per_hour=2.50,
        )
        text = format_cost_estimate(est)
        assert "H100" in text
        assert "large" in text
        assert "4.5" in text


# ---------------------------------------------------------------------------
# JSONL merge tests
# ---------------------------------------------------------------------------


class TestMergeJsonl:
    def test_merge_into_empty_file(self, tmp_path: Path) -> None:
        """Merging into a non-existent file creates it with all entries."""
        log_path = tmp_path / "experiments" / "log.jsonl"
        lines = [
            json.dumps({"timestamp": "2026-01-01T00:00:00", "metric": 0.5}),
            json.dumps({"timestamp": "2026-01-01T01:00:00", "metric": 0.6}),
        ]
        _merge_jsonl(lines, log_path)

        assert log_path.exists()
        written = log_path.read_text().strip().split("\n")
        assert len(written) == 2

    def test_deduplicates_by_timestamp(self, tmp_path: Path) -> None:
        """Entries with identical timestamps should not be duplicated."""
        log_path = tmp_path / "log.jsonl"
        existing = json.dumps({"timestamp": "2026-01-01T00:00:00", "metric": 0.5})
        log_path.write_text(existing + "\n")

        new_lines = [
            json.dumps({"timestamp": "2026-01-01T00:00:00", "metric": 0.5}),  # duplicate
            json.dumps({"timestamp": "2026-01-01T02:00:00", "metric": 0.7}),  # new
        ]
        _merge_jsonl(new_lines, log_path)

        entries = log_path.read_text().strip().split("\n")
        assert len(entries) == 2

    def test_handles_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines in remote output should be silently skipped."""
        log_path = tmp_path / "log.jsonl"
        lines = ["", json.dumps({"timestamp": "t1", "v": 1}), "", ""]
        _merge_jsonl(lines, log_path)

        entries = log_path.read_text().strip().split("\n")
        assert len(entries) == 1

    def test_handles_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON lines should be skipped without crashing."""
        log_path = tmp_path / "log.jsonl"
        lines = [
            "not valid json",
            json.dumps({"timestamp": "t1", "v": 1}),
            "{broken",
        ]
        _merge_jsonl(lines, log_path)

        entries = log_path.read_text().strip().split("\n")
        assert len(entries) == 1

    def test_preserves_existing_entries(self, tmp_path: Path) -> None:
        """Existing entries in the local file must not be overwritten."""
        log_path = tmp_path / "log.jsonl"
        existing = json.dumps({"timestamp": "t_local", "source": "local"})
        log_path.write_text(existing + "\n")

        remote = [json.dumps({"timestamp": "t_remote", "source": "remote"})]
        _merge_jsonl(remote, log_path)

        entries = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(entries) == 2
        assert entries[0]["source"] == "local"
        assert entries[1]["source"] == "remote"

    def test_entries_without_timestamp_are_skipped(self, tmp_path: Path) -> None:
        """Entries missing a timestamp field should be skipped (can't dedup)."""
        log_path = tmp_path / "log.jsonl"
        lines = [
            json.dumps({"no_timestamp": True}),
            json.dumps({"timestamp": "t1", "v": 1}),
        ]
        _merge_jsonl(lines, log_path)

        entries = log_path.read_text().strip().split("\n")
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Config size detection
# ---------------------------------------------------------------------------


class TestDetectConfigSize:
    def test_standard_config_paths(self) -> None:
        assert _detect_config_size("configs/tiny.toml") == "tiny"
        assert _detect_config_size("configs/small.toml") == "small"
        assert _detect_config_size("configs/medium.toml") == "medium"
        assert _detect_config_size("configs/large.toml") == "large"

    def test_custom_config_name(self) -> None:
        assert _detect_config_size("configs/my_experiment.toml") == "my_experiment"

    def test_nested_path(self) -> None:
        assert _detect_config_size("some/deep/path/tiny.toml") == "tiny"


# ---------------------------------------------------------------------------
# Provider registry tests
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_list_providers_returns_known_names(self) -> None:
        providers = list_providers()
        assert "runpod" in providers
        assert "vastai" in providers

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_error_message_lists_available(self) -> None:
        """The error message should list available providers for discoverability."""
        with pytest.raises(ValueError, match="runpod"):
            get_provider("aws")

    @patch("importlib.import_module")
    def test_lazy_import_called_with_correct_module(self, mock_import: MagicMock) -> None:
        """Registry should import the correct module path for each provider."""
        mock_module = MagicMock()
        mock_module.create_provider.return_value = FakeProvider()
        mock_import.return_value = mock_module

        provider = get_provider("runpod")
        mock_import.assert_called_once_with("chessgpt.cloud.providers.runpod")
        assert provider.name == "fake"

    @patch("importlib.import_module")
    def test_lazy_import_for_vastai(self, mock_import: MagicMock) -> None:
        mock_module = MagicMock()
        mock_module.create_provider.return_value = FakeProvider()
        mock_import.return_value = mock_module

        get_provider("vastai")
        mock_import.assert_called_once_with("chessgpt.cloud.providers.vastai")


# ---------------------------------------------------------------------------
# SSH helper tests
# ---------------------------------------------------------------------------


class TestSshExcludePatterns:
    def test_exact_match(self) -> None:
        from chessgpt.cloud.ssh import _should_exclude

        assert _should_exclude(Path(".git"), (".git",)) is True
        assert _should_exclude(Path("__pycache__"), ("__pycache__",)) is True

    def test_suffix_match(self) -> None:
        from chessgpt.cloud.ssh import _should_exclude

        assert _should_exclude(Path("game.pgn"), ("*.pgn",)) is True
        assert _should_exclude(Path("data.csv"), ("*.pgn",)) is False

    def test_no_match(self) -> None:
        from chessgpt.cloud.ssh import _should_exclude

        assert _should_exclude(Path("src"), (".git", "__pycache__")) is False

    def test_zst_suffix(self) -> None:
        from chessgpt.cloud.ssh import _should_exclude

        assert _should_exclude(Path("data.pgn.zst"), ("*.pgn.zst",)) is True

    def test_default_patterns_exclude_common_dirs(self) -> None:
        from chessgpt.cloud.ssh import DEFAULT_EXCLUDE_PATTERNS, _should_exclude

        assert _should_exclude(Path(".git"), DEFAULT_EXCLUDE_PATTERNS) is True
        assert _should_exclude(Path("__pycache__"), DEFAULT_EXCLUDE_PATTERNS) is True
        assert _should_exclude(Path("out"), DEFAULT_EXCLUDE_PATTERNS) is True
        assert _should_exclude(Path(".venv"), DEFAULT_EXCLUDE_PATTERNS) is True

    def test_default_patterns_allow_source(self) -> None:
        from chessgpt.cloud.ssh import DEFAULT_EXCLUDE_PATTERNS, _should_exclude

        assert _should_exclude(Path("src"), DEFAULT_EXCLUDE_PATTERNS) is False
        assert _should_exclude(Path("configs"), DEFAULT_EXCLUDE_PATTERNS) is False
        assert _should_exclude(Path("train.py"), DEFAULT_EXCLUDE_PATTERNS) is False


class TestSshMkdirP:
    def test_creates_nested_directories(self) -> None:
        """_sftp_mkdir_p should create each directory component."""
        from chessgpt.cloud.ssh import _sftp_mkdir_p

        mock_sftp = MagicMock()
        mock_sftp.stat.side_effect = FileNotFoundError

        _sftp_mkdir_p(mock_sftp, "/root/chessgpt/src")

        # Should attempt to create /root, /root/chessgpt, /root/chessgpt/src
        mkdir_calls = mock_sftp.mkdir.call_args_list
        paths = [c[0][0] for c in mkdir_calls]
        assert "/root" in paths
        assert "/root/chessgpt" in paths
        assert "/root/chessgpt/src" in paths

    def test_skips_existing_directories(self) -> None:
        """If a directory already exists (stat succeeds), don't create it."""
        from chessgpt.cloud.ssh import _sftp_mkdir_p

        mock_sftp = MagicMock()
        # stat succeeds for all directories (they exist)
        mock_sftp.stat.return_value = MagicMock()

        _sftp_mkdir_p(mock_sftp, "/root/chessgpt")

        mock_sftp.mkdir.assert_not_called()


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCliParsing:
    def _parse(self, args: list[str]) -> object:
        """Parse CLI args using the real parser from cloud.py."""
        from chessgpt.cli.cloud import _build_parser

        return _build_parser().parse_args(args)

    def test_train_args(self) -> None:
        args = self._parse(
            [
                "train",
                "--provider",
                "runpod",
                "--gpu",
                "A100",
                "--config",
                "configs/large.toml",
                "--name",
                "large_v1",
            ]
        )
        assert args.command == "train"
        assert args.provider == "runpod"
        assert args.gpu == "A100"
        assert args.config == "configs/large.toml"
        assert args.name == "large_v1"
        assert args.gpu_count == 1
        assert args.disk_gb == 50

    def test_eval_args(self) -> None:
        args = self._parse(
            [
                "eval",
                "--provider",
                "vastai",
                "--gpu",
                "RTX_4090",
                "--name",
                "small_v2",
            ]
        )
        assert args.command == "eval"
        assert args.provider == "vastai"
        assert args.name == "small_v2"

    def test_list_gpus_args(self) -> None:
        args = self._parse(["list-gpus", "--provider", "runpod"])
        assert args.command == "list-gpus"
        assert args.provider == "runpod"
        assert args.gpu is None

    def test_list_gpus_with_filter(self) -> None:
        args = self._parse(["list-gpus", "--provider", "runpod", "--gpu", "A100"])
        assert args.gpu == "A100"

    def test_custom_disk_and_gpu_count(self) -> None:
        args = self._parse(
            [
                "train",
                "--provider",
                "runpod",
                "--gpu",
                "H100",
                "--config",
                "configs/large.toml",
                "--name",
                "exp",
                "--gpu-count",
                "4",
                "--disk-gb",
                "200",
            ]
        )
        assert args.gpu_count == 4
        assert args.disk_gb == 200

    def test_missing_required_args_raises(self) -> None:
        """Missing --provider should cause a SystemExit."""
        with pytest.raises(SystemExit):
            self._parse(["train", "--gpu", "A100"])

    def test_status_subcommand(self) -> None:
        args = self._parse(["status"])
        assert args.command == "status"

    def test_attach_subcommand(self) -> None:
        args = self._parse(["attach"])
        assert args.command == "attach"

    def test_download_subcommand(self) -> None:
        args = self._parse(["download"])
        assert args.command == "download"
        assert args.output_dir == "out"

    def test_download_custom_output_dir(self) -> None:
        args = self._parse(["download", "--output-dir", "results"])
        assert args.output_dir == "results"

    def test_deprovision_subcommand(self) -> None:
        args = self._parse(["deprovision"])
        assert args.command == "deprovision"
        assert args.no_download is False

    def test_deprovision_no_download(self) -> None:
        args = self._parse(["deprovision", "--no-download"])
        assert args.no_download is True

    def test_deprovision_custom_output_dir(self) -> None:
        args = self._parse(["deprovision", "--output-dir", "results"])
        assert args.output_dir == "results"


# ---------------------------------------------------------------------------
# Pod state persistence tests
# ---------------------------------------------------------------------------


class TestPodState:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saving and loading should produce identical PodState."""
        from chessgpt.cloud.state import PodState, load_state, save_state

        pod = PodState(
            instance_id="pod-123",
            host="10.0.0.1",
            port=22222,
            username="root",
            ssh_key_path="~/.ssh/id_ed25519",
            gpu_type="A100",
            cost_per_hour=1.60,
            experiment_name="medium_v1",
            config_path="configs/medium.toml",
            provider_name="runpod",
            started_at="2026-01-15T10:00:00+00:00",
        )

        save_state(pod, tmp_path)
        loaded = load_state(tmp_path)

        assert loaded is not None
        assert loaded.instance_id == pod.instance_id
        assert loaded.host == pod.host
        assert loaded.port == pod.port
        assert loaded.experiment_name == pod.experiment_name
        assert loaded.config_path == pod.config_path
        assert loaded.provider_name == pod.provider_name
        assert loaded.started_at == pod.started_at
        assert loaded.tmux_session == "chessgpt-train"
        assert loaded.cost_per_hour == 1.60

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        from chessgpt.cloud.state import load_state

        assert load_state(tmp_path) is None

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        from chessgpt.cloud.state import PodState, clear_state, load_state, save_state

        pod = PodState(
            instance_id="pod-x",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-01T00:00:00+00:00",
        )
        save_state(pod, tmp_path)
        assert load_state(tmp_path) is not None

        clear_state(tmp_path)
        assert load_state(tmp_path) is None

    def test_clear_nonexistent_is_safe(self, tmp_path: Path) -> None:
        """Clearing when no state exists should not raise."""
        from chessgpt.cloud.state import clear_state

        clear_state(tmp_path)  # no error

    def test_from_instance_info(self, sample_info: InstanceInfo) -> None:
        from chessgpt.cloud.state import PodState

        pod = PodState.from_instance_info(
            sample_info,
            experiment_name="exp1",
            config_path="configs/large.toml",
            provider_name="runpod",
            started_at="2026-02-01T12:00:00+00:00",
        )

        assert pod.instance_id == sample_info.instance_id
        assert pod.host == sample_info.host
        assert pod.port == sample_info.port
        assert pod.gpu_type == sample_info.gpu_type
        assert pod.cost_per_hour == sample_info.cost_per_hour
        assert pod.experiment_name == "exp1"
        assert pod.provider_name == "runpod"

    def test_to_instance_info(self) -> None:
        from chessgpt.cloud.state import PodState

        pod = PodState(
            instance_id="pod-abc",
            host="1.2.3.4",
            port=22,
            username="root",
            ssh_key_path="/key",
            gpu_type="H100",
            cost_per_hour=2.50,
            experiment_name="test",
            config_path="configs/tiny.toml",
            provider_name="vastai",
            started_at="2026-01-01T00:00:00+00:00",
        )

        info = pod.to_instance_info()
        assert isinstance(info, InstanceInfo)
        assert info.instance_id == "pod-abc"
        assert info.host == "1.2.3.4"
        assert info.gpu_type == "H100"
        assert info.cost_per_hour == 2.50

    def test_save_creates_cloud_dir(self, tmp_path: Path) -> None:
        """save_state should create .cloud/ if it doesn't exist."""
        from chessgpt.cloud.state import PodState, save_state

        pod = PodState(
            instance_id="p",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="e",
            config_path="c",
            provider_name="runpod",
            started_at="t",
        )

        path = save_state(pod, tmp_path)
        assert path.exists()
        assert (tmp_path / ".cloud").is_dir()


# ---------------------------------------------------------------------------
# Orchestration integration tests (mocked SSH + provider)
# ---------------------------------------------------------------------------


class TestRunCloudTrainTmux:
    """Integration tests for the tmux-based run_cloud_train orchestrator.

    Verifies: provision → connect → upload → install → tmux launch → state saved.
    Pod is NOT deprovisioned on success (stays running for attach/status/download).
    """

    @patch("chessgpt.cloud.runner.save_state")
    @patch("chessgpt.cloud.runner.load_state", return_value=None)
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner._resolve_data_files")
    def test_tmux_launch_and_state_saved(
        self,
        mock_resolve: MagicMock,
        mock_ssh: MagicMock,
        mock_load: MagicMock,
        mock_save: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify tmux command is sent and state is saved."""
        from chessgpt.cloud.runner import run_cloud_train

        provider = FakeProvider()
        mock_resolve.return_value = []
        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client
        mock_ssh.upload_directory.return_value = 10
        mock_ssh.run_command.return_value = (0, "", "")

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "d.py").write_text("")
        (tmp_path / "configs").mkdir()
        (tmp_path / "pyproject.toml").write_text("")

        run_cloud_train(
            config_path="configs/tiny.toml",
            experiment_name="test_run",
            provider=provider,
            gpu_type="A100",
            project_root=tmp_path,
        )

        # Provision was called
        assert len(provider.provisioned) == 1

        # SSH commands: install deps, install tmux, launch tmux
        assert mock_ssh.run_command.call_count == 3

        # tmux command should be the third call
        tmux_call = mock_ssh.run_command.call_args_list[2]
        tmux_cmd_arg = tmux_call[0][1]  # positional arg: command string
        assert "tmux new-session -d -s chessgpt-train" in tmux_cmd_arg

        # State was saved
        mock_save.assert_called_once()

        # Pod was NOT deprovisioned (stays running)
        assert len(provider.deprovisioned) == 0

    @patch("chessgpt.cloud.runner.load_state", return_value=None)
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner._resolve_data_files")
    def test_deprovision_on_setup_failure(
        self,
        mock_resolve: MagicMock,
        mock_ssh: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If setup fails (e.g. install), the pod must be deprovisioned."""
        from chessgpt.cloud.runner import run_cloud_train

        provider = FakeProvider()
        mock_resolve.return_value = []
        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client
        mock_ssh.upload_directory.return_value = 5
        mock_ssh.run_command.side_effect = RuntimeError("Install failed")

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "d.py").write_text("")
        (tmp_path / "configs").mkdir()
        (tmp_path / "pyproject.toml").write_text("")

        with pytest.raises(RuntimeError, match="Install failed"):
            run_cloud_train(
                config_path="configs/tiny.toml",
                experiment_name="fail_run",
                provider=provider,
                gpu_type="A100",
                project_root=tmp_path,
            )

        # Pod must be deprovisioned on failure
        assert "fake-001" in provider.deprovisioned

    @patch("chessgpt.cloud.runner.load_state")
    def test_blocks_when_pod_already_active(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should raise if a pod is already active."""
        from chessgpt.cloud.runner import run_cloud_train
        from chessgpt.cloud.state import PodState

        mock_load.return_value = PodState(
            instance_id="existing-pod",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="old_exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-01T00:00:00+00:00",
        )

        provider = FakeProvider()

        with pytest.raises(RuntimeError, match="already active"):
            run_cloud_train(
                config_path="configs/tiny.toml",
                experiment_name="new_exp",
                provider=provider,
                gpu_type="A100",
                project_root=tmp_path,
            )

        # Nothing should have been provisioned
        assert len(provider.provisioned) == 0

    @patch("chessgpt.cloud.runner.load_state", return_value=None)
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner._resolve_data_files")
    def test_deprovision_on_connect_failure(
        self,
        mock_resolve: MagicMock,
        mock_ssh: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If SSH connect fails, deprovision should still be called."""
        from chessgpt.cloud.runner import run_cloud_train

        provider = FakeProvider()
        mock_resolve.return_value = []
        mock_ssh.connect.side_effect = ConnectionError("SSH failed")

        (tmp_path / "src").mkdir()
        (tmp_path / "configs").mkdir()
        (tmp_path / "pyproject.toml").write_text("")

        with pytest.raises(ConnectionError):
            run_cloud_train(
                config_path="configs/tiny.toml",
                experiment_name="conn_fail",
                provider=provider,
                gpu_type="A100",
                project_root=tmp_path,
            )

        assert "fake-001" in provider.deprovisioned


# ---------------------------------------------------------------------------
# Cloud status tests
# ---------------------------------------------------------------------------


class TestCloudStatus:
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner.load_state")
    def test_running_status(self, mock_load: MagicMock, mock_ssh: MagicMock) -> None:
        """Status should detect training is running when no sentinel exists."""
        from chessgpt.cloud.runner import cloud_status
        from chessgpt.cloud.state import PodState

        mock_load.return_value = PodState(
            instance_id="pod-1",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-15T10:00:00+00:00",
        )

        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client

        # cat exit_code fails (no sentinel) = still running
        # tail log succeeds
        mock_ssh.run_command.side_effect = [
            RuntimeError("No such file"),  # cat .train_exit_code
            (0, "Epoch 3/10 loss=0.5\n", ""),  # tail log
        ]

        cloud_status()  # should not raise

        # SSH was connected and closed
        mock_ssh.connect.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner.load_state")
    def test_completed_status(self, mock_load: MagicMock, mock_ssh: MagicMock) -> None:
        """Status should detect training completed via sentinel file."""
        from chessgpt.cloud.runner import cloud_status
        from chessgpt.cloud.state import PodState

        mock_load.return_value = PodState(
            instance_id="pod-1",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-15T10:00:00+00:00",
        )

        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client

        # cat exit_code returns "0" = done
        # tail log succeeds
        mock_ssh.run_command.side_effect = [
            (0, "0\n", ""),  # cat .train_exit_code
            (0, "Training complete\n", ""),  # tail log
        ]

        cloud_status()  # should not raise

    @patch("chessgpt.cloud.runner.load_state", return_value=None)
    def test_no_active_pod(self, mock_load: MagicMock) -> None:
        """Status with no active pod should print message, not raise."""
        from chessgpt.cloud.runner import cloud_status

        cloud_status()  # should not raise


# ---------------------------------------------------------------------------
# Cloud deprovision tests
# ---------------------------------------------------------------------------


class TestCloudDeprovision:
    @patch("chessgpt.cloud.runner.clear_state")
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner.load_state")
    def test_downloads_terminates_clears(
        self, mock_load: MagicMock, mock_ssh: MagicMock, mock_clear: MagicMock
    ) -> None:
        """Deprovision should download, terminate, and clear state."""
        from chessgpt.cloud.runner import cloud_deprovision
        from chessgpt.cloud.state import PodState

        mock_load.return_value = PodState(
            instance_id="pod-1",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-15T10:00:00+00:00",
        )

        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client
        mock_ssh.download_directory.return_value = 3
        mock_ssh.run_command.return_value = (0, "", "")

        provider = FakeProvider()
        cloud_deprovision(provider=provider)

        # Download attempted
        mock_ssh.download_directory.assert_called_once()

        # Pod terminated
        assert "pod-1" in provider.deprovisioned

        # State cleared
        mock_clear.assert_called_once()

    @patch("chessgpt.cloud.runner.clear_state")
    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner.load_state")
    def test_no_download_flag(
        self, mock_load: MagicMock, mock_ssh: MagicMock, mock_clear: MagicMock
    ) -> None:
        """With download=False, should skip download step."""
        from chessgpt.cloud.runner import cloud_deprovision
        from chessgpt.cloud.state import PodState

        mock_load.return_value = PodState(
            instance_id="pod-1",
            host="h",
            port=22,
            username="root",
            ssh_key_path="/k",
            gpu_type="A100",
            cost_per_hour=1.0,
            experiment_name="exp",
            config_path="configs/tiny.toml",
            provider_name="runpod",
            started_at="2026-01-15T10:00:00+00:00",
        )

        provider = FakeProvider()
        cloud_deprovision(download=False, provider=provider)

        # No SSH connection for download
        mock_ssh.connect.assert_not_called()

        # Still deprovisioned
        assert "pod-1" in provider.deprovisioned
        mock_clear.assert_called_once()

    @patch("chessgpt.cloud.runner.load_state", return_value=None)
    def test_no_active_pod(self, mock_load: MagicMock) -> None:
        """Deprovision with no active pod should print message, not raise."""
        from chessgpt.cloud.runner import cloud_deprovision

        cloud_deprovision()  # should not raise


# ---------------------------------------------------------------------------
# Cloud eval tests (unchanged behavior)
# ---------------------------------------------------------------------------


class TestRunCloudEval:
    """Integration tests for the run_cloud_eval orchestrator."""

    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner._resolve_data_files")
    def test_eval_requires_existing_model(
        self, mock_resolve: MagicMock, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """Eval should fail fast if no model.pt exists locally."""
        from chessgpt.cloud.runner import run_cloud_eval

        provider = FakeProvider()

        with pytest.raises(FileNotFoundError, match="No model found"):
            run_cloud_eval(
                experiment_name="missing",
                provider=provider,
                gpu_type="A100",
                output_dir=str(tmp_path / "out"),
                project_root=tmp_path,
            )

        # No provisioning should have happened
        assert len(provider.provisioned) == 0

    @patch("chessgpt.cloud.runner.ssh")
    @patch("chessgpt.cloud.runner._resolve_data_files")
    def test_eval_full_lifecycle(
        self, mock_resolve: MagicMock, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """Verify eval provisions, connects, uploads model, runs eval, deprovisions."""
        from chessgpt.cloud.runner import run_cloud_eval

        provider = FakeProvider()
        mock_resolve.return_value = []

        mock_client = MagicMock()
        mock_ssh.connect.return_value = mock_client
        mock_ssh.upload_directory.return_value = 3
        mock_ssh.download_directory.return_value = 1
        mock_ssh.run_command.return_value = (0, "", "")

        # Create model directory with model.pt
        model_dir = tmp_path / "out" / "test_eval"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(b"fake model")

        # Create minimal project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "dummy.py").write_text("")
        (tmp_path / "configs").mkdir()
        (tmp_path / "pyproject.toml").write_text("")

        run_cloud_eval(
            experiment_name="test_eval",
            provider=provider,
            gpu_type="A100",
            output_dir=str(tmp_path / "out"),
            project_root=tmp_path,
        )

        assert len(provider.provisioned) == 1
        assert "fake-001" in provider.deprovisioned


# ---------------------------------------------------------------------------
# SSH connect retry tests (with mocked paramiko)
# ---------------------------------------------------------------------------


class TestSshConnect:
    @patch("chessgpt.cloud.ssh.paramiko.SSHClient")
    @patch("chessgpt.cloud.ssh.time.sleep")
    def test_connect_succeeds_on_first_try(
        self, mock_sleep: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        from chessgpt.cloud.ssh import connect

        mock_client = mock_client_cls.return_value
        mock_client.connect.return_value = None

        client = connect("host", 22, "root", "/key", retries=3, initial_delay=0.1)

        assert client is mock_client
        mock_sleep.assert_not_called()

    @patch("chessgpt.cloud.ssh.paramiko.SSHClient")
    @patch("chessgpt.cloud.ssh.time.sleep")
    def test_connect_retries_on_failure(
        self, mock_sleep: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """Should retry on SSH exceptions and succeed when connection becomes available."""
        import paramiko

        from chessgpt.cloud.ssh import connect

        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = [
            paramiko.SSHException("not ready"),
            paramiko.SSHException("still not ready"),
            None,  # success on third attempt
        ]

        client = connect("host", 22, "root", "/key", retries=5, initial_delay=0.1)
        assert client is mock_client
        assert mock_sleep.call_count == 2

    @patch("chessgpt.cloud.ssh.paramiko.SSHClient")
    @patch("chessgpt.cloud.ssh.time.sleep")
    def test_connect_raises_after_exhausted_retries(
        self, mock_sleep: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        import paramiko

        from chessgpt.cloud.ssh import connect

        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = paramiko.SSHException("never ready")

        with pytest.raises(ConnectionError, match="Failed to connect.*after 3 attempts"):
            connect("host", 22, "root", "/key", retries=3, initial_delay=0.01)

    @patch("chessgpt.cloud.ssh.paramiko.SSHClient")
    @patch("chessgpt.cloud.ssh.time.sleep")
    def test_connect_handles_os_errors(
        self, mock_sleep: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        """OSError (e.g. connection refused) should also trigger retries."""
        from chessgpt.cloud.ssh import connect

        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = [
            OSError("Connection refused"),
            None,
        ]

        client = connect("host", 22, "root", "/key", retries=3, initial_delay=0.01)
        assert client is mock_client


# ---------------------------------------------------------------------------
# SSH run_command tests
# ---------------------------------------------------------------------------


class TestSshRunCommand:
    def test_raises_on_nonzero_exit(self) -> None:
        """Non-zero exit code should raise RuntimeError."""
        from chessgpt.cloud.ssh import run_command

        mock_client = MagicMock()
        mock_transport = MagicMock()
        mock_client.get_transport.return_value = mock_transport

        mock_channel = MagicMock()
        mock_transport.open_session.return_value = mock_channel

        # Channel returns no data and exits with code 1
        mock_channel.recv_ready.return_value = False
        mock_channel.recv_stderr_ready.return_value = False
        mock_channel.exit_status_ready.side_effect = [False, True]
        mock_channel.recv_exit_status.return_value = 1
        # Final drain calls
        mock_channel.recv.return_value = b""

        with pytest.raises(RuntimeError, match="Command failed"):
            run_command(mock_client, "false")

    def test_raises_on_no_transport(self) -> None:
        """If the SSH transport is None, should raise RuntimeError."""
        from chessgpt.cloud.ssh import run_command

        mock_client = MagicMock()
        mock_client.get_transport.return_value = None

        with pytest.raises(RuntimeError, match="transport is not active"):
            run_command(mock_client, "ls")

    def test_successful_command_returns_output(self) -> None:
        """A zero-exit command should return (0, stdout, stderr)."""
        from chessgpt.cloud.ssh import run_command

        mock_client = MagicMock()
        mock_transport = MagicMock()
        mock_client.get_transport.return_value = mock_transport

        mock_channel = MagicMock()
        mock_transport.open_session.return_value = mock_channel

        # Simulate: one chunk of stdout in the main loop, then exit.
        # recv_ready is called in: (1) main loop data check, (2) main loop sleep guard,
        # (3) drain loop. We need enough True/False entries to cover all paths.
        mock_channel.exit_status_ready.side_effect = [False, True]
        mock_channel.recv_ready.side_effect = [True, False, False, False]
        mock_channel.recv_stderr_ready.side_effect = [False, False, False, False]
        mock_channel.recv.return_value = b"hello world\n"
        mock_channel.recv_exit_status.return_value = 0

        code, stdout, stderr = run_command(mock_client, "echo hello", stream=False)
        assert code == 0
        assert "hello world" in stdout
        assert stderr == ""


# ---------------------------------------------------------------------------
# SSH upload/download tests (with mocked SFTP)
# ---------------------------------------------------------------------------


class TestSshUploadDirectory:
    def test_uploads_files_and_creates_dirs(self, tmp_path: Path) -> None:
        """upload_directory should create remote dirs and upload regular files."""
        from chessgpt.cloud.ssh import upload_directory

        # Create a local directory tree
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "file1.py").write_text("code")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "file2.py").write_text("more code")
        (tmp_path / "root.txt").write_text("root file")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError  # all dirs need creating

        count = upload_directory(mock_client, tmp_path, "/remote", exclude=())

        assert count == 3  # 3 files uploaded
        # Verify sftp.put was called for each file
        put_calls = mock_sftp.put.call_args_list
        assert len(put_calls) == 3

    def test_excludes_patterns(self, tmp_path: Path) -> None:
        """Files matching exclude patterns should be skipped."""
        from chessgpt.cloud.ssh import upload_directory

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython.pyc").write_text("bytecode")
        (tmp_path / "data.pgn").write_text("pgn data")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError

        count = upload_directory(mock_client, tmp_path, "/remote", exclude=("__pycache__", "*.pgn"))

        # Only src/main.py should be uploaded
        assert count == 1

    def test_sftp_closed_on_error(self, tmp_path: Path) -> None:
        """SFTP connection should be closed even if an error occurs during upload."""
        from chessgpt.cloud.ssh import upload_directory

        (tmp_path / "file.txt").write_text("data")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError
        mock_sftp.put.side_effect = IOError("upload failed")

        with pytest.raises(IOError):
            upload_directory(mock_client, tmp_path, "/remote", exclude=())

        mock_sftp.close.assert_called_once()


class TestSshDownloadDirectory:
    def test_downloads_files(self, tmp_path: Path) -> None:
        """download_directory should fetch remote files to local."""
        import stat as stat_mod

        from chessgpt.cloud.ssh import download_directory

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        # Simulate a remote directory with one file
        mock_entry = MagicMock()
        mock_entry.filename = "model.pt"
        mock_entry.st_mode = stat_mod.S_IFREG | 0o644  # regular file
        mock_sftp.listdir_attr.return_value = [mock_entry]

        count = download_directory(mock_client, "/remote/out", tmp_path / "local")

        assert count == 1
        mock_sftp.get.assert_called_once()

    def test_recurses_into_subdirs(self, tmp_path: Path) -> None:
        """download_directory should recursively descend into subdirectories."""
        import stat as stat_mod

        from chessgpt.cloud.ssh import download_directory

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp

        # Simulate: top-level has a subdir, subdir has a file
        dir_entry = MagicMock()
        dir_entry.filename = "subdir"
        dir_entry.st_mode = stat_mod.S_IFDIR | 0o755

        file_entry = MagicMock()
        file_entry.filename = "data.csv"
        file_entry.st_mode = stat_mod.S_IFREG | 0o644

        mock_sftp.listdir_attr.side_effect = [
            [dir_entry],  # top-level
            [file_entry],  # inside subdir
        ]

        count = download_directory(mock_client, "/remote", tmp_path / "local")

        assert count == 1  # one file total
        mock_sftp.get.assert_called_once()

    def test_sftp_closed_on_error(self, tmp_path: Path) -> None:
        """SFTP should be closed even if listing fails."""
        from chessgpt.cloud.ssh import download_directory

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.listdir_attr.side_effect = IOError("SFTP error")

        with pytest.raises(IOError):
            download_directory(mock_client, "/remote", tmp_path / "local")

        mock_sftp.close.assert_called_once()


class TestSshUploadFile:
    def test_uploads_single_file(self, tmp_path: Path) -> None:
        """upload_file should create parent dirs and upload the file."""
        from chessgpt.cloud.ssh import upload_file

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError

        upload_file(mock_client, local_file, "/remote/dir/test.txt")

        mock_sftp.put.assert_called_once_with(str(local_file), "/remote/dir/test.txt")
        mock_sftp.close.assert_called_once()

    def test_sftp_closed_on_error(self, tmp_path: Path) -> None:
        """SFTP should be closed even if put() fails."""
        from chessgpt.cloud.ssh import upload_file

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")

        mock_client = MagicMock()
        mock_sftp = MagicMock()
        mock_client.open_sftp.return_value = mock_sftp
        mock_sftp.stat.side_effect = FileNotFoundError
        mock_sftp.put.side_effect = IOError("failed")

        with pytest.raises(IOError):
            upload_file(mock_client, local_file, "/remote/test.txt")

        mock_sftp.close.assert_called_once()


# ---------------------------------------------------------------------------
# GpuOffer extra dict isolation
# ---------------------------------------------------------------------------


class TestGpuOfferIsolation:
    def test_extra_dicts_are_independent(self) -> None:
        """Each GpuOffer should have its own extra dict (no shared mutable default)."""
        a = GpuOffer(gpu_type="A", gpu_count=1, cost_per_hour=1.0)
        b = GpuOffer(gpu_type="B", gpu_count=1, cost_per_hour=2.0)
        a.extra["key"] = "value"
        assert "key" not in b.extra
