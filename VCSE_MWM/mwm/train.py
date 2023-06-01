import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
from keras import backend as K
import ruamel.yaml as yaml

import agent as agent
import common


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)
    logdir = f'logs/task={config.task}_use_vcse={config.use_vcse}_beta={config.beta_init}_seed={config.seed}'
    logdir = pathlib.Path(logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)
    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec

        prec.set_policy(prec.Policy("mixed_float16"))

    train_replay = common.Replay(logdir / "train_episodes", **config.replay)
    eval_replay = common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length,
        ),
    )
    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_train_mae = common.Every(config.train_mae_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)

    def make_env(mode):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = common.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera
            )
            env = common.NormalizeAction(env)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = common.MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                config.render_size,
                config.camera,
            )
            env = common.NormalizeAction(env)
        elif suite == "rlbench":
            env = common.RLBench(
                task,
                config.render_size,
                config.action_repeat,
            )
            env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode, store=None):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "metaworld" in config.task or "rlbench" in config.task:
            success = float(np.sum(ep["success"]) >= 1.0)
            print(
                f"{mode.title()} episode has {float(success)} success, {length} steps and return {score:.1f}."
            )
            logger.scalar(f"{mode}_success", float(success))
            if store != None :
                store['success'] += success
        else:
            print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        if store != None :
            store['return'] += score
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print("Create envs.")
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == "none":
        train_envs = [make_env("train") for _ in range(config.envs)]
        eval_envs = [make_env("eval") for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(
            functools.partial(make_env, mode), config.envs_parallel
        )
        train_envs = [make_async_env("train") for _ in range(config.envs)]
        eval_envs = [make_async_env("eval") for _ in range(num_eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs)
    store = {'success':0,'return':0}
    train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval" , store=store))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill:
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    mae_train_dataset = iter(train_replay.dataset(**config.mae_dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_mae = agnt.train_mae
    train_agent = common.CarryOverState(agnt.train)

    train_mae(next(mae_train_dataset))
    train_agent(next(train_dataset))
    
    if (logdir / "variables.pkl").exists():
        # load pre-trained RL agents parameters
        agnt.load(logdir / "variables.pkl")
    else:
        print("Pretrain agent.")
        for _ in range(config.mae_pretrain - config.pretrain):
            train_mae(next(mae_train_dataset))
        for _ in range(config.pretrain):
            train_mae(next(mae_train_dataset))
            train_agent(next(train_dataset))

    train_policy = lambda *args: agnt.policy(*args, mode="train")
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train_mae(step):
            for _ in range(config.train_mae_steps):
                mets = train_mae(next(mae_train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

    train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print("Start evaluation.")
        store['success'] = 0
        store['return'] = 0
        eval_driver(eval_policy, episodes=config.eval_eps)
        succ_ratio = store['success'] / config.eval_eps * 100
        return_mean = store['return'] / config.eval_eps
        logger.scalar(f"eval_success_ratio", float(succ_ratio))
        logger.scalar(f"eval_retrun_mean", float(return_mean))
        print("Start training.")
        train_driver(train_policy, steps=config.eval_every)
        agnt.save(logdir / "variables.pkl")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    agnt.save(logdir / "variables.pkl")


if __name__ == "__main__":
    main()
