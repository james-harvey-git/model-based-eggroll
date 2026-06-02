"""Unit tests for logger naming, tagging, and crash handling."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from mbrl.logger import Logger, _auto_name, _legend_fields, auto_tags, make_wm_group

BASE_CFG = {
    "seed": 0,
    "stage": "all",
    "dataset": {"name": "mujoco/halfcheetah/medium-v0"},
    "world_model": {
        "_target_": "mbrl.world_models.unifloral_ensemble_mlp.UnifloralEnsembleMLP"
    },
    "policy_optimizer": {"_target_": "mbrl.policy_optimizers.mopo.train"},
    "wandb": {"enabled": False, "tags": []},
}
TS = "20260410-143022"


class TestAutoName:
    def test_world_model_stage_excludes_algo(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "world_model"})
        name = _auto_name(cfg, TS)
        assert name == f"unifloral-halfcheetah-medium-s0-{TS}-wm"
        assert "mopo" not in name

    def test_policy_stage_includes_algo(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "policy"})
        name = _auto_name(cfg, TS)
        assert name == f"unifloral-mopo-halfcheetah-medium-s0-{TS}-pol"

    def test_all_stage_no_suffix(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "all"})
        name = _auto_name(cfg, TS)
        assert name == f"unifloral-mopo-halfcheetah-medium-s0-{TS}"


class TestMakeWmGroup:
    def test_format(self):
        cfg = OmegaConf.create(BASE_CFG)
        group = make_wm_group(cfg, TS)
        assert group == f"unifloral-halfcheetah-medium-s0-{TS}"
        assert "mopo" not in group


class TestAutoTags:
    def test_wm_stage_excludes_algo_tag(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "world_model"})
        tags = auto_tags(cfg)
        assert "unifloral" in tags
        assert "mopo" not in tags
        assert "halfcheetah-medium" in tags

    def test_policy_stage_includes_algo_tag(self):
        cfg = OmegaConf.create({**BASE_CFG, "stage": "policy"})
        tags = auto_tags(cfg)
        assert "unifloral" in tags
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
        cfg = OmegaConf.create({**BASE_CFG, "wandb": {"enabled": False, "tags": ["unifloral"]}})
        assert auto_tags(cfg).count("unifloral") == 1

    def test_finetune_tag_from_init_checkpoint(self):
        cfg = OmegaConf.create(
            {**BASE_CFG, "world_model": {
                "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
                "trainer": "eggroll",
                "init_checkpoint": "/tmp/some_ckpt.pkl",
            }}
        )
        assert "finetune" in auto_tags(cfg)

    def test_eggroll_indep_vs_shared_perturbation_tag(self):
        indep = OmegaConf.create(
            {**BASE_CFG, "world_model": {
                "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
                "trainer": "eggroll",
                "use_shared_perturbations": False,
            }}
        )
        shared = OmegaConf.create(
            {**BASE_CFG, "world_model": {
                "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
                "trainer": "eggroll",
                "use_shared_perturbations": True,
            }}
        )
        assert "indep-pert" in auto_tags(indep)
        assert "shared-pert" not in auto_tags(indep)
        assert "shared-pert" in auto_tags(shared)


class TestLegendFields:
    def test_unifloral_baseline_is_backprop_trainer(self):
        cfg = OmegaConf.create(BASE_CFG)
        fields = _legend_fields(cfg)
        assert fields["trainer"] == "backprop"
        assert fields["backbone"] == "mlp"
        # The old taxonomy fields are gone.
        assert "arch" not in fields
        assert "optimizer" not in fields

    def test_ensemble_mlp_backprop_trainer(self):
        cfg = OmegaConf.create(
            {**BASE_CFG, "world_model": {
                "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
                "trainer": "backprop",
                "num_ensemble": 7,
                "num_elites": 5,
                "batch_size": 256,
                "lr": 1e-3,
            }}
        )
        fields = _legend_fields(cfg)
        assert fields["trainer"] == "backprop"
        assert fields["num_ensemble"] == 7
        assert fields["num_elites"] == 5
        assert fields["batch_size"] == 256
        assert "population_size" not in fields

    def test_ensemble_mlp_eggroll_trainer_exposes_eggroll_fields(self):
        cfg = OmegaConf.create(
            {**BASE_CFG, "world_model": {
                "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
                "trainer": "eggroll",
                "backbone": "mlp",
                "num_ensemble": 7,
                "use_shared_perturbations": False,
                "eggroll": {
                    "lr": 0.05,
                    "population_size": 512,
                    "sigma": 0.02,
                    "group_size": 64,
                },
            }}
        )
        fields = _legend_fields(cfg)
        assert fields["trainer"] == "eggroll"
        assert fields["population_size"] == 512
        assert fields["sigma"] == 0.02
        assert fields["group_size"] == 64
        assert fields["use_shared_perturbations"] is False
        assert "batch_size" not in fields

    def test_returns_empty_without_world_model(self):
        cfg = OmegaConf.create({k: v for k, v in BASE_CFG.items() if k != "world_model"})
        assert _legend_fields(cfg) == {}


class TestLoggerWmGroup:
    def test_wm_group_stored_as_attribute(self):
        cfg = OmegaConf.create(BASE_CFG)
        logger = Logger(cfg, wm_group="unifloral-halfcheetah-medium-s0-20260410-143022")
        assert logger.wm_group == "unifloral-halfcheetah-medium-s0-20260410-143022"

    def test_wm_group_auto_generated_when_not_provided(self):
        cfg = OmegaConf.create(BASE_CFG)
        logger = Logger(cfg)
        assert logger.wm_group.startswith("unifloral-halfcheetah-medium-s0-")
        # Timestamp portion: YYYYMMDD-HHMMSS = 15 chars
        ts_part = logger.wm_group.split("-s0-", 1)[1]
        assert len(ts_part) == 15


class TestFromExistingRun:
    """from_existing_run (used by sweep agents) must apply the same legend fields,
    provenance, and default step metric as __init__ on the already-init'd run."""

    def _eggroll_cfg(self):
        return OmegaConf.create({**BASE_CFG, "world_model": {
            "_target_": "mbrl.world_models.ensemble_mlp.EnsembleMLP",
            "trainer": "eggroll", "num_ensemble": 7,
            "eggroll": {"lr": 3e-4, "population_size": 512, "sigma": 0.02, "group_size": 8},
        }})

    def test_applies_legend_and_metrics(self):
        with patch("mbrl.logger.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            logger = Logger.from_existing_run(self._eggroll_cfg(), wm_group="g")
            assert logger.enabled and logger.wm_group == "g"
            mock_wandb.config.update.assert_called_once()
            sent = mock_wandb.config.update.call_args[0][0]
            assert sent["trainer"] == "eggroll"
            assert sent["population_size"] == 512
            assert mock_wandb.define_metric.called

    def test_finetune_lineage_none_without_init_checkpoint(self):
        with patch("mbrl.logger.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            logger = Logger.from_existing_run(self._eggroll_cfg())
            assert logger.finetune_lineage is None


class TestLogPolicyStep:
    def test_does_not_force_global_step(self):
        """Policy logging must record policy/step as a field, not as W&B's global
        step — otherwise a stage=all run drops every policy row as non-monotonic
        against the world-model stage's millions-scale step."""
        logger = Logger.__new__(Logger)
        logger.enabled = True
        with patch("mbrl.logger.wandb") as mock_wandb:
            logger.log_policy_step(25000, critic_loss=0.5, raw_score=-100.0)
            mock_wandb.log.assert_called_once()
            args, kwargs = mock_wandb.log.call_args
            assert "step" not in kwargs  # global step not forced
            payload = args[0]
            assert payload["policy/step"] == 25000
            assert payload["policy/critic_loss"] == 0.5
            assert payload["policy/raw_score"] == -100.0


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
