import random
from argparse import Namespace
from typing import Callable

from absl import logging
import gym
import numpy as np
import torch
import pickle


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()
    observations = []
    actionss = []
    data = {
        'actions': [],      
        'observations': [], 
    }
    obs, _ = envs.reset()
    import ipdb; ipdb.set_trace()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        observations.append(obs)
        actionss.append(actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                logging.info(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
                data["actions"].append(actionss)
                data["observations"].append(observations)
                actionss = []
                observations = []
        obs = next_obs
    data["returns"] = episodic_returns
    filename = 'datasets/data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def evaluate_single(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = False,
):  
    env = make_env(env_id, 0, 0, capture_video, run_name)
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(env, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()
    data = {
        'actions': [],      
        'observations': [], 
        'returns': []       
    }
    
    ob, _ = env.reset()
    observations = []
    actions = []

    while len(data["returns"]) < eval_episodes:
        if random.random() < epsilon:
            #actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            action = np.random.randint(env.action_space.n)  
        else:
            action, _ = model.get_action(torch.Tensor(ob).to(device))
            action = action.cpu().numpy()
        next_ob, _, _, _, info = env.step(action)
        if "episode" in info:
\            logging.info(f"eval_episode={len(data['returns'])}, episodic_return={info['episode']['r']}")
    
            #import ipdb; ipdb.set_trace()
            data["observations"].append(observations)
            data["actions"].append(actions)
            data["returns"].append(info["episode"]["r"])
            observations = []
            actions = []
            ob, _ = env.reset()
        observations.append(ob)
        actions.append(action)
        ob = next_ob

    filename = 'datasets/data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    #from huggingface_hub import hf_hub_download

    from cleanrl.c51_atari import QNetwork, make_env

    model_path = "/scr/matthewh6/atari-representation-learning/runs/PongNoFrameskip-v4__c51_atari__1__1718755025/c51_atari.cleanrl_model"
    evaluate(
        model_path,
        make_env,
        "PongNoFrameskip-v4",
        eval_episodes=500,
        run_name=f"pong-eval-3",
        Model=QNetwork,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )