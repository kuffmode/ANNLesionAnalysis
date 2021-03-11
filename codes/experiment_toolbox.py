import numpy as np
import copy
import game_play_toolbox as gpto
import pandas as pd
import scipy.stats

def pruner(genome):
    for link in list(genome.connections.keys()):
        if genome.connections[link].enabled == False:
            genome.connections.pop(link)


def network_structure(config, genome):
    input_nodes = set()
    hidden_nodes = set()
    output_nodes = set()
    connections = set()
    weights = set()
    activationf = []
    aggregationf = []
    biaslist = []

    for k in config.genome_config.input_keys:
        input_nodes.add(k)

    for k in config.genome_config.output_keys:
        output_nodes.add(k)

    for cg in genome.connections.values():
        if cg.enabled:
            connections.add(cg.key)
            weights.add(float(cg.weight))

    for k in genome.nodes:
        hidden_nodes.add(k)
        temp = genome.nodes.get(k)
        activationf.append(temp.activation)
        aggregationf.append(temp.aggregation)
        biaslist.append(round(float(temp.bias), 6))
    hidden_nodes = list(set(hidden_nodes) ^ set(output_nodes))

    input_nodes = [key for key in input_nodes]
    output_nodes = [key for key in output_nodes]
    connections = [key for key in connections]
    weights = [key for key in weights]

    edges = list(zip(connections, weights))
    edges = [(i, j, w) for (i, j), w in edges]

    return input_nodes, hidden_nodes, output_nodes, connections, weights, edges, activationf, aggregationf, biaslist


def weight_swapper(original_genome, weight_vector):
    weights = copy.deepcopy(weight_vector)
    weight_swapped_genome = copy.deepcopy(original_genome)

    for connection in list(weight_swapped_genome.connections.keys()):
        random_weight = np.random.choice(weights, replace=False)
        weight_swapped_genome.connections[connection].weight = random_weight
        weights.remove(random_weight)

    return weight_swapped_genome


def weight_tracker(original_genome, weight_swapped_genome):
    tracker = {}
    for connection in original_genome.connections.keys():
        for corresponding_connection in weight_swapped_genome.connections.keys():

            if original_genome.connections[connection].weight \
                    == weight_swapped_genome.connections[corresponding_connection].weight:
                weight = original_genome.connections[connection].weight
                tracker[tuple((connection, corresponding_connection))] = weight
    return tracker


def connectivity_randomizer(original_genome, network_information, n_self_loops):
    connectivity_shuffled_genome = copy.deepcopy(original_genome)
    first = network_information['input_nodes'] + \
            network_information['hidden_nodes']
    second = network_information['output_nodes'] + \
             network_information['hidden_nodes']
    new_connections = {}
    connections = list(original_genome.connections.keys())

    for connection in connections:
        keys = tuple((np.random.choice(first),
                     np.random.choice(second)))
        while keys in new_connections.keys():
            keys = tuple((np.random.choice(first),
                         np.random.choice(second)))
        new_connections[keys] = copy.deepcopy(original_genome.connections[connection])

    new_links = list(new_connections.keys())
    for _ in range(n_self_loops):
        self_loop_one = (np.random.choice(second))
        self_loop = tuple((self_loop_one, self_loop_one))

        while self_loop in new_links:
            self_loop_one = (np.random.choice(second))
            self_loop = tuple((self_loop_one, self_loop_one))
        else:
            second.remove(self_loop_one)
            random_pick = np.random.choice(len(connections))
            new_connections[self_loop] = new_connections.pop(new_links[random_pick])
            new_links = list(new_connections.keys())

    connectivity_shuffled_genome.connections = new_connections

    for connection in connectivity_shuffled_genome.connections.keys():
        connectivity_shuffled_genome.connections[connection].key = connection

    return connectivity_shuffled_genome


def weight_binarizer(optimized_genome):
    weight_binarized_genome = copy.deepcopy(optimized_genome)
    for connection in weight_binarized_genome.connections.keys():
        weight_binarized_genome.connections[connection].weight =\
            int(np.random.choice([1,-1],1))

    return weight_binarized_genome


def single_lesion(genome, n_games, **params):
    dataset = {}
    links = [_ for _ in genome.connections]
    for link in links:
        genome.connections[link].enabled = False

        print(f'Perturbing link: {genome.connections[link]}')
        results = gpto.play_games(gpto.play_game,
                                  n_games=n_games,
                                  genome = genome,
                                  n_jobs=-1, **params)
        genome.connections[link].enabled = True
        dataset[link] = results
    dataset['intact'] = gpto.play_games(gpto.play_game,
                              n_games=n_games,
                              genome = genome,
                              n_jobs=-1, **params)
    return dataset


def preprocess_plot(data, reference_data=None):
    data = pd.DataFrame(data)
    data.columns = [str(c) for c in data.columns]
    if reference_data is not None:
        reference_data = pd.DataFrame(reference_data)
        reference_data.columns = [str(c) for c in reference_data.columns]
        data = data.reindex(reference_data.mean().sort_values().index, axis=1)
    else:
        data = data.reindex(data.mean().sort_values().index, axis=1)
    return data


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def bootstrap_hyp_test(data, p_value = 0.05, bootstrap_samples = 10000, reference_mean = None):

    mean_adjusted_distributions = pd.DataFrame()
    bootstrapped_distributions = pd.DataFrame()
    significants = pd.DataFrame()
    percentile = (1 - p_value)*100

    for distribution in data.columns:
        if reference_mean is not None:
            mean_adjusted_distributions[distribution] = data[distribution] - data[distribution].mean() + reference_mean
        else:
            mean_adjusted_distributions[distribution] = data[distribution] - data[distribution].mean()
        temp = []

        # resampling from the mean-adjusted distribution
        for sample in range(bootstrap_samples):
            temp.append(np.mean(
                (np.random.choice(list(mean_adjusted_distributions[distribution]),
                                  len(mean_adjusted_distributions[distribution].values),
                                  replace=True))))

        bootstrapped_distributions[distribution] = temp

    for i in bootstrapped_distributions.columns:
        percentiles = np.percentile(bootstrapped_distributions[i], [0, percentile])
        if percentiles[0] <= data[i].mean() <= percentiles[1]:
            pass
        else:
            significants[i] = data[i]

    significants = significants.reindex(significants.mean().sort_values().index, axis=1)
    return significants


def node_to_link(links, target_node,translated_links):
    for i in links:
        for j in i:
            if j == target_node:
                if i not in translated_links:
                    translated_links.append(i)
