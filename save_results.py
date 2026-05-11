import pickle
from pathlib import Path
from typing import Dict, Any, Optional



def save_results_to_pickle(result, base_dir='results_MNIST', custom_filename=None):
    run_params = result.get('run_params', {})
    num_agents = run_params.get('sys_num_agents', 'unknown')
    topology = run_params.get('algo_topology', 'unknown')
    degree = run_params.get('sys_degree', None)
    distribution = run_params.get('data_distribution', 'unknown')
    algorithm = run_params.get('algo_name', 'unknown')
    seed = run_params.get('seed', 'no_seed')
    
    if topology in ['circle', 'mesh'] and degree is not None:
        topology_name = f"{topology}_deg{degree}"
    else:
        topology_name = topology
    
    network_config = run_params.get('data_network_config')
    
    if network_config and not isinstance(network_config, str):
        parts = []
        
        # Потери пакетов
        if hasattr(network_config, 'packet_loss_rate') and network_config.packet_loss_rate > 0:
            parts.append(f"loss_{int(network_config.packet_loss_rate*100)}")
        
        # Шум
        if hasattr(network_config, 'noise_std') and network_config.noise_std > 0:
            parts.append(f"noise_{network_config.noise_std}")
        
        # Автоматические отключения
        if hasattr(network_config, 'node_dropout_prob') and network_config.node_dropout_prob > 0:
            dropout_mode = "stop" if network_config.dropout_stops_training else "continue"
            dropout_min = getattr(network_config, 'dropout_rounds_min', 1)
            dropout_max = getattr(network_config, 'dropout_rounds_max', 5)
            parts.append(f"auto_dropout_{int(network_config.node_dropout_prob*100)}_{dropout_mode}_{dropout_min}-{dropout_max}")
        
        # Задержки
        if hasattr(network_config, 'delay_max') and network_config.delay_max > 0:
            delay_min = getattr(network_config, 'delay_min', 0)
            delay_max = network_config.delay_max
            delay_prob = getattr(network_config, 'delay_probability', 0.3)
            parts.append(f"delay_{delay_min}-{delay_max}_{delay_prob}")
        
        # Ручные отключения
        if hasattr(network_config, 'scheduled_dropouts') and network_config.scheduled_dropouts:
            dropout_details = []
            for dropout in network_config.scheduled_dropouts:
                node_id = dropout.get('node_id', '?')
                start = dropout.get('start_round', '?')
                duration = dropout.get('duration', '?')
                dropout_details.append(f"n{node_id}s{start}d{duration}")
            train_mode = "train" if not network_config.dropout_stops_training else "notrain"
            if dropout_details:
                scheduled_str = f"scheduled_{'_'.join(dropout_details)}_{train_mode}"
                parts.append(scheduled_str)
        
        network_name = "_".join(parts) if parts else "ideal"
    else:
        network_name = "ideal"
    
    folder_path = Path(base_dir) / f"{num_agents}_agents" / topology_name / distribution / network_name / algorithm
    folder_path.mkdir(parents=True, exist_ok=True)
    
    if custom_filename:
        filename = f"{algorithm}_{seed}_{custom_filename}.pkl"
    else:
        filename = f"{algorithm}_seed{seed}.pkl"
    
    filepath = folder_path / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Результаты сохранены в: {filepath}")
    return str(filepath)

def load_results_from_pickle(filename):
    """
    Загружает результаты эксперимента.
    """
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    
    print(f"Результаты загружены из {filename}")
    return result