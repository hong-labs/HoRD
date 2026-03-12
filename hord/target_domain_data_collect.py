import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
simulator = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "simulator" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +simulator")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            simulator = "isaacgym"
        elif "isaaclab" in arg.split("=")[-1]:
            from isaaclab.app import AppLauncher

            simulator = "isaaclab"

        elif "genesis" in arg.split("=")[-1]:
            simulator = "genesis"

from lightning.fabric import Fabric  # noqa: E402
from hord.utils.config_utils import *  # noqa: E402, F403

from hord.agents.ppo.agent import PPO  # noqa: E402


@hydra.main(config_path="config")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True

        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                print(f"Could not find config path: {config_path}")

        if has_config:
            print(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()
    if simulator == "isaaclab":
        app_launcher = AppLauncher({"headless": config.headless})
        simulation_app = app_launcher.app
        env = instantiate(
            config.env, device=fabric.device, simulation_app=simulation_app
        )
    else:
        env = instantiate(config.env, device=fabric.device)

    agent: PPO = instantiate(config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(config.checkpoint)

    agent.target_domain_data_collection()


if __name__ == "__main__":
    main()

#python hord/target_domain_data_collect.py +exp=full_body_tracker/transformer_flat_terrain +robot=g1 +simulator=genesis ++headless=false +checkpoint=results/g1_walk_DR_action/last.ckpt ++motion_file=data/motions/g1_walk.npy ++num_envs=1 +terrain=flat
#python hord/eval_agent.py +base=[fabric,structure] +exp=deepmimic_mlp +robot=g1 +simulator=isaaclab +checkpoint=null +training_max_steps=1 +motion_file=data/collected_motions/g1_walk.npy env.config.sync_motion=True ref_respawn_offset=0 +headless=False num_envs=1  +experiment_name=debug +terrain=flat


