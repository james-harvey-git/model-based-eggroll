"""Unit tests for logger naming, tagging, and crash handling."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from mbrl.logger import Logger, _auto_name, auto_tags, make_wm_group

BASE_CFG = {
    "seed": 0,
    "stage": "all",
    "dataset": {"name": "mujoco/halfcheetah/medium-v0"},
    "world_model": {"_target_": "mbrl.world_models.mle.MLEEnsemble"},
    "policy_optimizer": {"_target_": "mbrl.policy_optimizers.mopo.train"},
    "wandb": {"enabled": False, "tags": []},
}
TS = "20260410-143022"


class TestAutoName:
    def test_world_model_stage_excludes_algo(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "world_model"})
        name = _auto_name(cfg, TS)
        assert name == f"mle-halfcheetah-medium-s0-{TS}-wm"
        assert "mopo" not in name

    def test_policy_stage_includes_algo(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "policy"})
        name = _auto_name(cfg, TS)
        assert name == f"mle-mopo-halfcheetah-medium-s0-{TS}-pol"

    def test_all_stage_no_suffix(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "all"})
        name = _auto_name(cfg, TS)
        assert name == f"mle-mopo-halfcheetah-medium-s0-{TS}"


class TestMakeWmGroup:
    def test_format(self):
        cfg = OmegaConf.create(BASE_CFG)
        group = make_wm_group(cfg, TS)
        assert group == f"mle-halfcheetah-medium-s0-{TS}"
        assert "mopo" not in group


class TestAutoTags:
    def test_wm_stage_excludes_algo_tag(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "world_model"})
        tags = auto_tags(cfg)
        assert "mle" in tags
        assert "mopo" not in tags
        assert "halfcheetah-medium" in tags

    def test_policy_stage_includes_algo_tag(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "policy"})
        tags = auto_tags(cfg)
        assert "mle" in tags
        assert "mopo" in tags

    def test_sweep_tag_from_env(self, monkeypatch):
        monkeypatch.setenv("WANDB_SWEEP_ID", "abc123")
        cfg = OmegaConf.create(BASE_CFG)
        assert "sweep" in auto_tags(cfg)

    def test_sweep_tag_from_slurm_launcher_env(self, monkeypatch):
        monkeypatch.setenv("SWEEP_ID", "abc123")
        cfg = OmegaConf.create(BASE_CFG)
        assert "sweep" in auto_tags(cfg)

    def test_sweep_tag_from_active_wandb_run(self):
        cfg = OmegaConf.create(BASE_CFG)
        with patch("mbrl.logger.wandb") as mock_wandb:
            mock_wandb.run = MagicMock(sweep_id="abc123")
            assert "sweep" in auto_tags(cfg)

    def test_cluster_tag_from_env(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        cfg = OmegaConf.create(BASE_CFG)
        assert "cluster" in auto_tags(cfg)

    def test_manual_tags_merged(self):
        cfg = OmegaConf.create({**BASE_CFG, "wandb": {"enabled": False, "tags": ["final"]}})
        assert "final" in auto_tags(cfg)

    def test_deduplication(self):
        cfg = OmegaConf.create({**BASE_CFG, "wandb": {"enabled": False, "tags": ["mle"]}})
        assert auto_tags(cfg).count("mle") == 1


class TestLoggerWmGroup:
    def test_wm_group_stored_as_attribute(self):
        cfg = OmegaConf.create(BASE_CFG)
        logger = Logger(cfg, wm_group="mle-halfcheetah-medium-s0-20260410-143022")
        assert logger.wm_group == "mle-halfcheetah-medium-s0-20260410-143022"

    def test_wm_group_auto_generated_when_not_provided(self):
        cfg = OmegaConf.create(BASE_CFG)
        logger = Logger(cfg)
        assert logger.wm_group.startswith("mle-halfcheetah-medium-s0-")
        # Timestamp portion: YYYYMMDD-HHMMSS = 15 chars
        ts_part = logger.wm_group.split("-s0-", 1)[1]
        assert len(ts_part) == 15


class TestSetCrashedTag:
    def test_noop_when_disabled(self):
        cfg = OmegaConf.create({**BASE_CFG, "wandb": {"enabled": False, "tags": []}})
        logger = Logger(cfg)
        logger.set_crashed_tag()  # must not raise

    def test_adds_crashed_tag(self):
        logger = Logger.__new__(Logger)
        logger.enabled = True
        with patch("mbrl.logger.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_run.tags = ("mopo",)
            mock_wandb.run = mock_run
            logger.set_crashed_tag()
            assert "crashed" in mock_run.tags

    def test_idempotent(self):
        logger = Logger.__new__(Logger)
        logger.enabled = True
        with patch("mbrl.logger.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_run.tags = ("crashed",)
            mock_wandb.run = mock_run
            logger.set_crashed_tag()
            logger.set_crashed_tag()
            assert mock_run.tags.count("crashed") == 1
