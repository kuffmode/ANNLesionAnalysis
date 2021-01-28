import pickle
import pandas as pd
import copy

genome_file = '/home/kayson/PycharmProjects/' \
              'shapleyvalue/network_3_fitness_1300_.pkl'

with open(genome_file, 'rb') as f:
    optimized_genome = pickle.load(f)


def placeholder_maker(genome):
    empty_template = pd.DataFrame(index=genome.connections.keys(),
                                  columns=genome.connections.keys())
    empty_placeholder = {'interactions': copy.deepcopy(empty_template),
                         'interactions_gamma_a': copy.deepcopy(empty_template),
                         'gamma_a': copy.deepcopy(empty_template)}
    return empty_placeholder


for link in list(optimized_genome.connections.keys()):
    if not optimized_genome.connections[link].enabled:
        optimized_genome.connections.pop(link)

results = placeholder_maker(optimized_genome)

n_trials = 100

for trial in range(n_trials):
    with open(f'results_{trial}.pkl', 'wb') as f:
        pickle.dump(results, f, 1)
