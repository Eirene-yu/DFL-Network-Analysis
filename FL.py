import random
from utils import init_metrics_dict, average_weights, log_current_metrics
from simulate_network import init_network_stats, manual_dropout, NetworkConditions, get_active_neighbors, update_node_status, simulate_network, simulate_network_pame
import numpy as np
import torch

    
def select_gossip_neighbors(all_neighbors, gossip_neighbors, rng, participation_rate=None):
    """
    Выбирает соседей для gossip обмена.
    """
    if not all_neighbors:
        return []
    
    # Случайный сосед
    if gossip_neighbors == 1 or gossip_neighbors == 'random':
        return [rng.choice(all_neighbors)]
    
    # Все соседи
    if gossip_neighbors == 'all':
        if participation_rate is not None and 0 < participation_rate <= 1:
            
            k = max(1, int(len(all_neighbors) * participation_rate))
            k = min(k, len(all_neighbors))
            if k == len(all_neighbors):
                return all_neighbors.copy()
            return rng.choice(all_neighbors, size=k, replace=False).tolist()
        return all_neighbors.copy()
    
    # Фиксированное количество соседей
    if isinstance(gossip_neighbors, int):
        k = min(gossip_neighbors, len(all_neighbors))
        return rng.choice(all_neighbors, size=k, replace=False).tolist()
    
    return [rng.choice(all_neighbors)]

def train_dpsgd(agents, topology, learning_rate=0.5, coeff_lr=0.99, num_rounds=100, gossip_neighbors='all', network_conditions=None, random_seed=42, test_loader=None, model_class=None, 
                return_all_metrics=False, eval_every=1, fix_learning_rate=False, verbose=True, return_agent_metrics=False, communication_period=1):
    """
    D-PSGD алгоритм.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_agents = len(agents)
    rng = np.random.RandomState(random_seed)
    
    # Инициализация метрик
    metric_dic = init_metrics_dict(return_all_metrics)
    
    # Веса агентов
    agent_sizes = [agent.get_sample_number() for agent in agents]
    total_data = sum(agent_sizes)
    agent_weights = [size / total_data for size in agent_sizes]
    
    if network_conditions is None:
        network_conditions = NetworkConditions()
    
    network_stats = init_network_stats(num_agents, network_conditions, random_seed)
    for dropout in network_conditions.scheduled_dropouts:
        manual_dropout(network_stats, dropout['node_id'], dropout['start_round'], dropout['duration'], verbose=verbose)
    
    # Настройка убывающего шага обучения
    if not fix_learning_rate:
        decay = coeff_lr if coeff_lr < 1 else 0.99
    
    metric_dic = log_current_metrics(
        agents=agents,
        metric_dic=metric_dic,
        test_loader=test_loader,
        model_class=model_class,
        agent_weights=agent_weights,
        return_all_metrics=return_all_metrics,
        return_agent_metrics=return_agent_metrics,
        network_stats=network_stats
    )
    
    for round_idx in range(num_rounds):
        if not fix_learning_rate:
            current_lr = learning_rate * (decay ** round_idx)
        else:
            current_lr = learning_rate
        
        # Определение активных агентов
        active_nodes = update_node_status(network_stats, round_idx, network_conditions, verbose=verbose)
        
        # Получение текущих весов
        current_weights = [agent.get_weights() for agent in agents]

        # Коммуникация
        should_communicate = (round_idx % communication_period == 0)

        if should_communicate:
            # Симуляция ненадежной сети
            received_weights, stats, network_stats = simulate_network(weights_list=current_weights, conditions=network_conditions, 
                                                                      network_stats=network_stats, current_round=round_idx, verbose=verbose)
            stats['active_nodes'] = active_nodes.copy()
            if verbose and len(received_weights) == 0:
                print(f"[D-PSGD] Round {round_idx+1}: Все пакеты потеряны/задержаны")
        else:
            received_weights = {i: current_weights[i] for i in range(num_agents) if active_nodes[i]}
            stats = {
                'lost': [],
                'delayed': [],
                'delivered_from_delay': [],
                'dropped': [i for i in range(num_agents) if not active_nodes[i]],
                'inactive': [],
                'received_count': sum(active_nodes),
                'total_nodes': num_agents,
                'round': round_idx,
                'pending_delays': len(network_stats.get('delayed_packets', [])),
                'still_waiting': [p['node_id'] for p in network_stats.get('delayed_packets', []) if p['delivery_round'] > round_idx],
                'active_nodes': active_nodes.copy()
            }
            if verbose and communication_period > 1:
                print(f"[D-PSGD] Round {round_idx+1}: Нет коммуникации (следующая в раунде {((round_idx // communication_period) + 1) * communication_period + 1})")
        
        # Усреднение параметров
        new_weights = [None] * num_agents
        
        for agent_idx, agent in enumerate(agents):
            if not active_nodes[agent_idx]:
                # Неактивный агент не участвует в усреднении
                new_weights[agent_idx] = agent.get_weights()
                continue
            
            all_neighbors = topology.get(agent_idx, [])
            # Учитываем только активных соседей
            active_neighbors = get_active_neighbors(network_stats, all_neighbors)
            selected_neighbors = select_gossip_neighbors(active_neighbors, gossip_neighbors, rng)

            if not selected_neighbors:
                new_weights[agent_idx] = current_weights[agent_idx]
                continue
            
            # Собираем веса для усреднения
            weights_to_avg = [current_weights[agent_idx]]
            weights_for_avg = [agent_weights[agent_idx]]
            
            for neighbor_idx in selected_neighbors:
                if neighbor_idx in received_weights and received_weights[neighbor_idx] is not None:
                    weights_to_avg.append(received_weights[neighbor_idx])
                    weights_for_avg.append(agent_weights[neighbor_idx])
            
            # Усреднение
            avg_weights = average_weights(weights_to_avg)
            
            if avg_weights is not None:
                new_weights[agent_idx] = avg_weights
            else:
                new_weights[agent_idx] = current_weights[agent_idx]
        
        # Применяем усредненные веса
        for agent_idx, agent in enumerate(agents):
            if new_weights[agent_idx] is not None:
                agent.set_weights(new_weights[agent_idx])
        
        # Локальный градиентный шаг
        for agent_idx, agent in enumerate(agents):
            should_train = False
            if active_nodes[agent_idx]:
                should_train = True
            else:
                if network_conditions.dropout_stops_training:
                    should_train = False
                else:
                    should_train = True

            if should_train:
                agent.train(current_lr)
        
        metric_dic = log_current_metrics(
            agents=agents,
            metric_dic=metric_dic,
            test_loader=test_loader,
            model_class=model_class,
            agent_weights=agent_weights,
            return_all_metrics=return_all_metrics,
            network_stats=stats,
            is_final_round=(round_idx == num_rounds - 1),
            return_agent_metrics=return_agent_metrics,
            active_nodes=active_nodes
        )
        
        if verbose and test_loader is not None and (round_idx % eval_every == 0 or round_idx == num_rounds - 1):
            print(f"[D-PSGD] Round {round_idx+1}: Acc={metric_dic['accuracy'][-1]:.4f}, Acc_active={metric_dic['accuracy_active'][-1]:.4f}, "
                  f"Loss={metric_dic['loss'][-1]:.4f}, Loss_active={metric_dic['loss_active'][-1]:.4f}, " 
                  f"Consensus={metric_dic['consensus_error'][-1]:.6f}, Consensus_active={metric_dic['consensus_error_active'][-1]:.6f}")
    
    # Финальная модель
    final_weights = average_weights([agent.get_weights() for agent in agents])
    
    global_model = None
    if final_weights is not None and model_class is not None:
        global_model = model_class()
        global_model.load_state_dict(final_weights)
    
    return global_model, metric_dic

def pme_average(received_messages, local_weights):
    """
    PME усреднение (Partial Message Exchange)
    """
    if not received_messages:
        return local_weights
    
    averaged = {}
    
    for key in local_weights.keys():
        shape = local_weights[key].shape
        num_params = local_weights[key].numel()
        accumulated = torch.zeros(num_params, dtype=torch.float32)
        lambda_counts = torch.zeros(num_params, dtype=torch.float32)
        
        for msg in received_messages:
            packed_data = msg['packed_data']
            if key in packed_data:
                layer_data = packed_data[key]
                indices = layer_data['indices']
                values = layer_data['values']
                
                if isinstance(indices, np.ndarray):
                    indices = torch.from_numpy(indices).long()
                if isinstance(values, np.ndarray):
                    values = torch.from_numpy(values).float()
                
                flat_sparse = torch.zeros(num_params, dtype=torch.float32)
                flat_sparse[indices] = values
                flat_mask = torch.zeros(num_params, dtype=torch.float32)
                flat_mask[indices] = 1.0

                accumulated += flat_sparse
                lambda_counts += flat_mask
        
        result_flat = torch.zeros(num_params, dtype=torch.float32)
        # хотя бы один сосед прислал эту координату
        mask_nonzero = lambda_counts > 0
        result_flat[mask_nonzero] = accumulated[mask_nonzero] / lambda_counts[mask_nonzero]
        # никто не прислал эту координату
        mask_zero = ~mask_nonzero
        if mask_zero.any():
            local_flat = local_weights[key].flatten().float()
            result_flat[mask_zero] = local_flat[mask_zero]
        
        averaged[key] = result_flat.reshape(shape)
    
    return averaged

def train_pame(agents, topology, num_rounds=100, gossip_neighbors='all', network_conditions=None, random_seed=42,
               test_loader=None, model_class=None, return_all_metrics=False, eval_every=1, verbose=True, return_agent_metrics=False, 
               communication_period=1, sparsification_rate=0.2, participation_rate=0.5, sigma_0=1.0, gamma=1.005):
    """
    PaME алгоритм.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_agents = len(agents)
    rng = np.random.RandomState(random_seed)
    
    # Инициализация метрик
    metric_dic = init_metrics_dict(return_all_metrics)
    
    # Веса агентов
    agent_sizes = [agent.get_sample_number() for agent in agents]
    total_data = sum(agent_sizes)
    agent_weights = [size / total_data for size in agent_sizes]
    
    sigma = [sigma_0] * num_agents
    prev_m_i = [0] * num_agents
    m_i = [0] * num_agents
    
    if network_conditions is None:
        network_conditions = NetworkConditions()
    
    network_stats = init_network_stats(num_agents, network_conditions, random_seed)
    for dropout in network_conditions.scheduled_dropouts:
        manual_dropout(network_stats, dropout['node_id'], dropout['start_round'], dropout['duration'], verbose=verbose)
    
    metric_dic = log_current_metrics(
        agents=agents,
        metric_dic=metric_dic,
        test_loader=test_loader,
        model_class=model_class,
        agent_weights=agent_weights,
        return_all_metrics=return_all_metrics,
        return_agent_metrics=return_agent_metrics,
        network_stats=network_stats
    )
    
    for round_idx in range(num_rounds):
        # Определение активных агентов
        active_nodes = update_node_status(network_stats, round_idx, network_conditions, verbose=verbose)
        
        # Получение текущих весов
        current_weights = [agent.get_weights() for agent in agents]
        
        # Коммуникация
        should_communicate = (round_idx % communication_period == 0)
        
        messages_to_send = {}
        if should_communicate:
            for sender_id, agent in enumerate(agents):
                if not active_nodes[sender_id]:
                    continue
                
                all_neighbors = topology.get(sender_id, [])
                active_neighbors = get_active_neighbors(network_stats, all_neighbors)
                selected_neighbors = select_gossip_neighbors(active_neighbors, gossip_neighbors, rng, participation_rate)

                m_i[sender_id] = len(selected_neighbors)
                prev_m_i[sender_id] = m_i[sender_id]
                
                # Сообщения для каждого выбранного соседа
                for recipient_id in selected_neighbors:
                    if recipient_id not in messages_to_send:
                        messages_to_send[recipient_id] = []
                    messages_to_send[recipient_id].append((sender_id, current_weights[sender_id]))
        else:
            for agent_idx in range(num_agents):
                m_i[agent_idx] = prev_m_i[agent_idx]
        
        # Симуляция ненадежной сети
        if should_communicate and messages_to_send:
            received_messages, stats, network_stats = simulate_network_pame(messages_to_send=messages_to_send, conditions=network_conditions, network_stats=network_stats,
                current_round=round_idx, sparsification_rate=sparsification_rate, base_seed=random_seed, verbose=verbose)
            stats['active_nodes'] = active_nodes.copy()
            if verbose and len(received_messages) == 0:
                print(f"[PaME] Round {round_idx+1}: Все пакеты потеряны/задержаны")
        else:
            received_messages = {}
            stats = {
                'lost': [], 'delayed': [], 'delivered_from_delay': [],
                'dropped': [i for i in range(num_agents) if not active_nodes[i]],
                'received_count': 0,
                'total_nodes': num_agents,
                'round': round_idx,
                'pending_delays': len(network_stats.get('delayed_packets', [])),
                'still_waiting': [p.get('recipient_id', p.get('node_id')) for p in network_stats.get('delayed_packets', []) 
                                 if p.get('delivery_round', 0) > round_idx],
                'active_nodes': active_nodes.copy()
            }
            if verbose and not should_communicate and communication_period > 1:
                print(f"[PaME] Round {round_idx+1}: Нет коммуникации")
        
        # Усреднение параметров
        new_weights = [None] * num_agents
        
        for agent_idx, agent in enumerate(agents):
            if not active_nodes[agent_idx]:
                new_weights[agent_idx] = current_weights[agent_idx]
                continue
            
            if should_communicate and agent_idx in received_messages:
                new_weights[agent_idx] = pme_average(received_messages[agent_idx], current_weights[agent_idx])
            else:
                new_weights[agent_idx] = current_weights[agent_idx]
        
        # Применяем усредненные веса
        for agent_idx, agent in enumerate(agents):
            if new_weights[agent_idx] is not None:
                agent.set_weights(new_weights[agent_idx])
        
        # Локальный градиентный шаг
        for agent_idx, agent in enumerate(agents):
            should_train = False
            if active_nodes[agent_idx]:
                should_train = True
            else:
                if network_conditions.dropout_stops_training:
                    should_train = False
                else:
                    should_train = True

            if should_train:
                current_lr = 1.0 / (sigma[agent_idx] * max(1, m_i[agent_idx]))
                agent.train(current_lr)
        
        # Обновление штрафного параметра
        for agent_idx in range(num_agents):
            sigma[agent_idx] *= gamma
        
        metric_dic = log_current_metrics(
            agents=agents,
            metric_dic=metric_dic,
            test_loader=test_loader,
            model_class=model_class,
            agent_weights=agent_weights,
            return_all_metrics=return_all_metrics,
            network_stats=stats,
            is_final_round=(round_idx == num_rounds - 1),
            return_agent_metrics=return_agent_metrics,
            active_nodes=active_nodes
        )
        
        if verbose and test_loader is not None and (round_idx % eval_every == 0 or round_idx == num_rounds - 1):
            print(f"[PaME] Round {round_idx+1}: Acc={metric_dic['accuracy'][-1]:.4f}, Acc_active={metric_dic['accuracy_active'][-1]:.4f}, "
                  f"Loss={metric_dic['loss'][-1]:.4f}, Loss_active={metric_dic['loss_active'][-1]:.4f}, " 
                  f"Consensus={metric_dic['consensus_error'][-1]:.6f}, Consensus_active={metric_dic['consensus_error_active'][-1]:.6f}, "
                  f"σ={sigma[0]:.3f}, m={m_i[0]}")
    
    # Финальная модель
    final_weights = average_weights([agent.get_weights() for agent in agents])
    global_model = None
    if final_weights is not None and model_class is not None:
        global_model = model_class()
        global_model.load_state_dict(final_weights)
    
    return global_model, metric_dic
