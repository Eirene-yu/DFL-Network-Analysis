import random
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import Subset

def split_data(dataset, num_agents, distribution='iid', alpha=0.5, random_seed=42, verbose=False):
    """
    Разделяет датасет между агентами согласно заданному распределению.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    total_size = len(dataset)

    if distribution == 'iid':
        class_indices = list(range(total_size))
        random.shuffle(class_indices)

        agent_indices = [[] for _ in range(num_agents)]

        for i, idx in enumerate(class_indices):
            agent_indices[i % num_agents].append(idx)
            
    elif distribution == 'dirichlet':
        labels = [dataset[i][1] for i in range(total_size)]
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        agent_indices = [[] for _ in range(num_agents)]
        
        for class_id in sorted(class_indices.keys()):
            indices = class_indices[class_id].copy()
            np.random.shuffle(indices)
            
            proportions = np.random.dirichlet([alpha] * num_agents)
            split_points = (proportions * len(indices)).astype(int).cumsum()[:-1]
            splits = np.split(np.array(indices), split_points)
            
            for agent_idx, split in enumerate(splits):
                if len(split) > 0:
                    agent_indices[agent_idx].extend(split.tolist())
        
        for i in range(num_agents):
            random.shuffle(agent_indices[i])
    else:
        raise ValueError(f"Неизвестный тип распределения: {distribution}.")
    
    all_assigned_indices = set()
    for idxs in agent_indices:
        all_assigned_indices.update(idxs)
    
    if len(all_assigned_indices) != total_size:
        missing = set(range(total_size)) - all_assigned_indices
        for i, idx in enumerate(missing):
            agent_indices[i % num_agents].append(idx)
    
    if verbose:
        print("\nРаспределение данных по агентам:")
        total_used = 0
        for i, idxs in enumerate(agent_indices):
            if idxs:
                labels = [dataset[idx][1] for idx in idxs]
                class_counts = Counter(labels)
                agent_total = len(idxs)
                total_used += agent_total
                sorted_counts = dict(sorted(class_counts.items()))
                print(f"Агент {i} ({agent_total:5d} примеров): {sorted_counts}")
            else:
                print(f"Агент {i}: Нет данных!")
        
        print(f"Всего использовано: {total_used}/{total_size} примеров")
    return [Subset(dataset, idxs) for idxs in agent_indices]