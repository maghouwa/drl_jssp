from jssp.agent_utils import *
from jssp.jssp_env import SJSSP
from jssp.ppo_jssp import PPO
import torch
import numpy as np
import time


device = "cpu"

with open(os.path.join(os.path.dirname(__file__), "jssp", "config.json"), "r") as config_file:
    configs = json.load(config_file)

with open(os.path.join(os.path.dirname(__file__), "input_config_benchmark.json"), "r") as params_file:
    params = json.load(params_file)

N_JOBS_P = params["Pn_j"]
N_MACHINES_P = params["Pn_m"]
benchmark = params["which_benchmark"]
N_JOBS_N = params["Nn_j"]
N_MACHINES_N = params["Nn_m"]
LOW = configs["low"]
HIGH = configs["high"]


env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

ppo = PPO(configs["lr"], configs["gamma"], configs["k_epochs"], configs["eps_clip"],
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs["num_layers"],
          neighbor_pooling_type=configs["neighbor_pooling_type"],
          input_dim=configs["input_dim"],
          hidden_dim=configs["hidden_dim"],
          num_mlp_layers_feature_extract=configs["num_mlp_layers_feature_extract"],
          num_mlp_layers_actor=configs["num_mlp_layers_actor"],
          hidden_dim_actor=configs["hidden_dim_actor"],
          num_mlp_layers_critic=configs["num_mlp_layers_critic"],
          hidden_dim_critic=configs["hidden_dim_critic"])
path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
ppo.policy.load_state_dict(torch.load(path))
g_pool_step = g_pool_cal(graph_pool_type=configs["graph_pool_type"],
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)

dataLoaded = np.load('./BenchDataNmpy/' + benchmark + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.npy')
dataset = []
for i in range(dataLoaded.shape[0]):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1]))

result = []
t1 = time.time()
for i, data in enumerate(dataset):
    adj, fea, candidate, mask = env.reset(data)
    ep_reward = - env.max_endTime
    while True:
        # Running policy_old:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=candidate_tensor.unsqueeze(0),
                               mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, omega)
            action = greedy_select_action(pi, candidate)

        adj, fea, reward, done, candidate, mask = env.step(action)
        ep_reward += reward

        if done:
            break
    # print(max(env.end_time))
    print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
    result.append(-ep_reward + env.posRewards)
t2 = time.time()
file_writing_obj = open('./' + 'drltime_' + benchmark + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.txt', 'w')
file_writing_obj.write(str((t2 - t1)/len(dataset)))

# print(result)
# print(np.array(result, dtype=np.single).mean())
np.save('drlResult_' + benchmark + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P), np.array(result, dtype=np.single))
