import networkx as nx
import random

def create_topology(num_agents, degree, random_seed=42, max_attempts=50):
    """
    Создает регулярный связный граф.
    """
    if degree >= num_agents:
        degree = num_agents - 1
    
    if degree == 1:
        if num_agents == 2:
            G = nx.complete_graph(num_agents)
        else:
            raise ValueError(f"degree должен быть >= 2")
    
    for attempt in range(max_attempts):
        seed = random_seed + attempt * 100
        G = nx.random_regular_graph(degree, num_agents, seed=seed)
        if nx.is_connected(G):
            degrees = list(dict(G.degree()).values())
            if len(set(degrees)) == 1:
                topology = {n: list(G.neighbors(n)) for n in G.nodes()}
                return topology