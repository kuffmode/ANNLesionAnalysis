# -----------------------------------------------------------
# importing and preparing things
# -----------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
import networkx as nx
from game_play_toolbox import *
import os
import pickle
import tensorflow as tf

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'GothamSSm'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config_file = '/home/kayson/PycharmProjects/shapleyvalue/AEconfig-SI.txt'
genome_file = '/home/kayson/PycharmProjects/shapleyvalue/network_3_fitness_1300_.pkl'
ae_file = '/home/kayson/PycharmProjects/shapleyvalue/SI-AE-model.pkl'
config = neat.Config(neat.DefaultGenome,
                     neat.DefaultReproduction,
                     neat.DefaultSpeciesSet,
                     neat.DefaultStagnation,
                     config_file)

# Genome is basically the NEAT-optimized network
with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)

# NEAT produces junk (disabled) connections, this line prunes them
exto.pruner(optimized_genome)

intact_genome = copy.deepcopy(optimized_genome)

zeze_lesioned_genome = copy.deepcopy(optimized_genome)
mifoze_lesioned_genome = copy.deepcopy(optimized_genome)

zeze_lesioned_genome.connections[(0, 0)].enabled = False
mifoze_lesioned_genome.connections[(-4, 0)].enabled = False

both_lesioned_genome = copy.deepcopy(zeze_lesioned_genome)
both_lesioned_genome.connections[(-4, 0)].enabled = False

params = dict(environment_name='SpaceInvaders-v4',
              model_filename=ae_file,
              config=config)
input_nodes, hidden_nodes, output_nodes, links \
    , _, _, _, _, _ = exto.network_structure(config, optimized_genome)

def bulk_recorder(n_trials=50, min_samples=1_000, random_seed=0, genome=None, **params):
    timeseries = {}
    while len(timeseries) < n_trials:
        _, neural_data, _ = neural_recorder(genome=genome, env_seed=random_seed, **params)
        if len(neural_data) > min_samples:
            actions = action_argmax_mask(neural_data, input_nodes, hidden_nodes)
            timeseries[random_seed] = neural_data + actions
            print(f'Trials left: {n_trials - len(timeseries)}')
        random_seed += 1
    return timeseries


n_trials = 50
min_samples = 1200
random_seed = 0

intact_timeseries = bulk_recorder(min_samples = min_samples,n_trials=n_trials,genome=intact_genome,**params)
zeze_timeseries = bulk_recorder(min_samples = min_samples,n_trials=n_trials,genome=zeze_lesioned_genome,**params)
mifoze_timeseries = bulk_recorder(min_samples = min_samples,n_trials=n_trials,genome=mifoze_lesioned_genome,**params)
both_timeseries = bulk_recorder(min_samples = min_samples,n_trials=n_trials,genome=both_lesioned_genome,**params)

intact_idtxl = np.zeros((min_samples, 19, n_trials))
zeze_idtxl = np.zeros((min_samples, 19, n_trials))
mifoze_idtxl = np.zeros((min_samples, 19, n_trials))
both_idtxl = np.zeros((min_samples, 19, n_trials))

for idx, timeserie in enumerate(intact_timeseries):
    intact_idtxl[:, :, idx] = intact_timeseries[timeserie][:min_samples][:]
for idx, timeserie in enumerate(zeze_timeseries):
    zeze_idtxl[:, :, idx] = zeze_timeseries[timeserie][:min_samples][:]
for idx, timeserie in enumerate(mifoze_timeseries):
    mifoze_idtxl[:, :, idx] = mifoze_timeseries[timeserie][:min_samples][:]
for idx, timeserie in enumerate(both_timeseries):
    both_idtxl[:, :, idx] = both_timeseries[timeserie][:min_samples][:]

with open(f'intact_idtxl.pkl', 'wb') as f:
    pickle.dump(intact_idtxl, f, 1)
with open(f'zeze_idtxl.pkl', 'wb') as f:
    pickle.dump(zeze_idtxl, f, 1)
with open(f'mifoze_idtxl.pkl', 'wb') as f:
    pickle.dump(mifoze_idtxl, f, 1)
with open(f'both_idtxl.pkl', 'wb') as f:
    pickle.dump(both_idtxl, f, 1)