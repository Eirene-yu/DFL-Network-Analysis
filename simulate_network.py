import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class NetworkConditions:
    """
    Параметры ненадежности сети для децентрализованных алгоритмов
    """
    random_seed: int = None
    # Вероятность потери пакета
    packet_loss_rate: float = 0.0
    # Стандартное отклонение шума, добавляемого к передаваемым данным
    noise_std: float = 0.0 
    # Вероятность отключения узла из сети
    node_dropout_prob: float = 0.0
    # Минимальное количество раундов, на которое агент отключается
    dropout_rounds_min: int = 1
    # Максимальное количество раундов, на которое узел отключается
    dropout_rounds_max: int = 5       
    # True: узел полностью прекращает обучение
    # False: узел продолжает локальное обучение в момент отключения
    dropout_stops_training: bool = False
    # е (запланированные) отключения
    scheduled_dropouts: List[Dict] = field(default_factory=list)
    # Минимальная задержка при передаче (в раундах)
    delay_min: int = 0
    # Максимальная задержка при передаче (в раундах)
    delay_max: int = 0
    # Вероятность того, что пакет будет задержан (при delay_max > 0)
    delay_probability: float = 0.3

def get_active_nodes(state):
    """
    Возвращает массив активных узлов
    """
    return state['active_nodes'].copy()

def get_active_neighbors(state, all_neighbors):
    """
    Возвращает список активных соседей
    """
    return [j for j in all_neighbors if j < state['num_nodes'] and state['active_nodes'][j]]

def init_network_stats(num_nodes, conditions=None, random_seed=42):
    """
    Инициализация состояния сети
    """
    if conditions is not None and conditions.random_seed is not None:
        random_seed = conditions.random_seed
    rng = np.random.RandomState(random_seed)
    return {
        'num_nodes': num_nodes,
        'active_nodes': np.ones(num_nodes, dtype=bool),
        'node_dropout_counter': np.zeros(num_nodes, dtype=int),
        'dropout_history': [],
        'scheduled_dropouts': [],
        'delayed_packets': [],
        'rng': rng
    }


def manual_dropout(state, node_id, start_round, dropout_rounds, verbose=False):
    """
    Запланированное отключение узла в определенном раунде
    """
    if node_id >= state['num_nodes']:
        if verbose:
            print(f"Ошибка: узла {node_id} не существует (всего {state['num_nodes']} узлов)")
        return False
    
    if 'scheduled_dropouts' not in state:
        state['scheduled_dropouts'] = []
    
    state['scheduled_dropouts'].append({'node_id': node_id, 'start_round': start_round, 'duration': dropout_rounds})
    if verbose:
        print(f"Запланировано отключение узла {node_id} с раунда {start_round} на {dropout_rounds} раундов")
    return True

def apply_scheduled_dropouts(state, current_round, verbose=False):
    """
    Применяет запланированные отключения в нужном раунде
    """
    if 'scheduled_dropouts' not in state:
        return
    
    to_remove = []
    for i, schedule in enumerate(state['scheduled_dropouts']):
        # Если наступил раунд отключения
        if schedule['start_round'] == current_round:
            node_id = schedule['node_id']
            duration = schedule['duration']

            if state['node_dropout_counter'][node_id] == 0:
                state['node_dropout_counter'][node_id] = duration
                state['active_nodes'][node_id] = False
                state['dropout_history'].append((current_round, node_id, duration, 'manual'))
                if verbose:
                    print(f"Запланированное отключение: Узел {node_id} отключен на {duration} раундов (раунд {current_round})")
            else:
                if verbose:
                    print(f"Узел {node_id} уже отключен, игнорируем")
            to_remove.append(i)
    
    for i in reversed(to_remove):
        state['scheduled_dropouts'].pop(i)

def update_node_status(state, current_round, conditions, verbose=False):
    """
    Обновляет статус выключенных узлов
    """
    # Применяем запланированные отключения
    apply_scheduled_dropouts(state, current_round)

    for i in range(state['num_nodes']):
        # Агент отключен
        if state['node_dropout_counter'][i] > 0:
            state['node_dropout_counter'][i] -= 1
            if state['node_dropout_counter'][i] == 0:
                state['active_nodes'][i] = True
                if verbose:
                    print(f"Узел {i} вернулся в строй (раунд {current_round})")
        # Автоматическое отключение
        elif state['rng'].random() < conditions.node_dropout_prob:
            dropout_rounds = state['rng'].randint(conditions.dropout_rounds_min, conditions.dropout_rounds_max + 1)
            state['node_dropout_counter'][i] = dropout_rounds
            state['active_nodes'][i] = False
            state['dropout_history'].append((current_round, i, dropout_rounds))
            if verbose:
                print(f"Автоотключение: Узел {i} отключен на {dropout_rounds} раундов")

    return state['active_nodes'].copy()

def simulate_network(weights_list, conditions, network_stats, current_round, verbose=False):
    """
    Симуляция передачи пакетов
    """
    num_nodes = len(weights_list)

    active_nodes = network_stats['active_nodes']
    rng = network_stats['rng']
    
    received = {}
    delivered_from_delay = []
    still_delayed = []
    lost_agents = []
    delayed_agents = []
    dropped_agents = []

    if 'packet_counter' not in network_stats:
        network_stats['packet_counter'] = 0
    
    for packet in network_stats.get('delayed_packets', []):
        node_id = packet['node_id']
        packet_id = packet.get('packet_id', None)

        if packet['delivery_round'] <= current_round:
            if active_nodes[node_id]:
                received[node_id] = packet['weights']
                delivered_from_delay.append({'node_id': node_id, 'packet_id': packet_id})
                if verbose:
                    print(f"Пакет {packet_id} доставлен узлу {node_id} (раунд {current_round})")
            else:
                packet['delivery_round'] = current_round + 1
                packet['extra_delay'] = packet.get('extra_delay', 0) + 1
                delayed_agents.append({'node_id': node_id, 'delay': 1, 'reason': 'node_inactive', 'packet_id': packet_id, 'extra_delay_so_far': packet['extra_delay']})
                still_delayed.append(packet)
                if verbose:
                    print(f"Пакет {packet_id} для узла {node_id} отложен: получатель не активен (раунд {current_round})")
        else:
            still_delayed.append(packet)
    
    network_stats['delayed_packets'] = still_delayed
    
    for i, w in enumerate(weights_list):
        if i in received:
            continue
        
        if not active_nodes[i]:
            dropped_agents.append(i)
            continue
        
        # Потеря пакетов
        if rng.random() < conditions.packet_loss_rate:
            lost_agents.append(i)
            continue
        
        # Задержка
        if conditions.delay_max > 0 and rng.random() < conditions.delay_probability:
            delay = rng.randint(conditions.delay_min, conditions.delay_max + 1)
            if delay > 0:
                network_stats['packet_counter'] += 1
                packet_id = network_stats['packet_counter']
                network_stats['delayed_packets'].append({
                    'packet_id': packet_id, 'node_id': i, 'weights': w, 'delay': delay, 
                    'original_delay': delay, 'extra_delay': 0, 'delivery_round': current_round + delay
                })
                delayed_agents.append({
                    'node_id': i, 'delay': delay, 'reason': 'network',
                    'packet_id': packet_id, 'source_round': current_round
                })
                continue
        
        # Шум/ошибки передачи
        if conditions.noise_std > 0 and w is not None:
            noisy_w = {}
            for k, v in w.items():
                if any(pattern in k for pattern in ['running_mean', 'running_var', 'num_batches_tracked']):
                    noisy_w[k] = v
                else:
                    noise = torch.randn_like(v) * conditions.noise_std
                    noisy_w[k] = v + noise
            w = noisy_w

        received[i] = w
    
    stats = {
        'lost': lost_agents,
        'delayed': delayed_agents,
        'delivered_from_delay': delivered_from_delay,
        'dropped': dropped_agents,
        'received_count': len(received),
        'total_nodes': num_nodes,
        'round': current_round,
        'pending_delays': len(network_stats['delayed_packets']),
        'still_waiting': [p['node_id'] for p in network_stats['delayed_packets'] if p['delivery_round'] > current_round]
    }
     
    return received, stats, network_stats

def pack_sparse_message(sparse_weights, mask):
    """
    Разреженные данные для передачи в PaME
    """
    packed = {}
    for key in sparse_weights.keys():
        flat_mask = mask[key].flatten()
        indices = torch.where(flat_mask > 0)[0]
        flat_values = sparse_weights[key].flatten()
        values = flat_values[indices].float()
        
        packed[key] = {'indices': indices.cpu().numpy(), 'values': values.cpu().numpy(), 'shape': sparse_weights[key].shape}
    return packed

def sparsify_weights(weights, rate, base_seed, sender_id, recipient_id, current_round):
    """
    Разреживание весов для каждой пары отправитель и получатель.
    """
    unique_seed = hash((base_seed, sender_id, recipient_id, current_round))
    
    sparse = {}
    mask = {}
    
    for key, tensor in weights.items():
        num_params = tensor.numel()
        k = max(1, int(num_params * rate))
        
        flat = tensor.flatten().float()
        original_seed = torch.initial_seed()
        torch.manual_seed(unique_seed)
        indices = torch.randperm(num_params)[:k]
        torch.manual_seed(original_seed)
        
        sparse_flat = torch.zeros(num_params, dtype=torch.float32)
        sparse_flat[indices] = flat[indices]
        
        mask_flat = torch.zeros(num_params, dtype=torch.float32)
        mask_flat[indices] = 1.0
        
        sparse[key] = sparse_flat.reshape(tensor.shape)
        mask[key] = mask_flat.reshape(tensor.shape)
    
    return sparse, mask

def simulate_network_pame(messages_to_send, conditions, network_stats, current_round, sparsification_rate=0.2, base_seed=42, verbose=False):
    """
    Симуляция передачи для PaME
    """
    active_nodes = network_stats['active_nodes']
    rng = network_stats['rng']
    
    received = {}
    delivered_from_delay = []
    still_delayed = []
    lost_messages = []
    delayed_messages = []
    dropped_agents = []

    for packet in network_stats.get('delayed_packets', []):
        recipient_id = packet['recipient_id']
        if packet['delivery_round'] <= current_round:
            if active_nodes[recipient_id]:
                if recipient_id not in received:
                    received[recipient_id] = []
                received[recipient_id].append({'packed_data': packet['packed_data'], 'sender_id': packet['sender_id']})
                delivered_from_delay.append({'node_id': recipient_id, 'packet_id': packet['packet_id'], 'sender_id': packet['sender_id']})
                if verbose:
                    print(f"Пакет от {packet['sender_id']} доставлен узлу {recipient_id} (раунд {current_round})")
            else:
                packet['delivery_round'] = current_round + 1
                packet['extra_delay'] = packet.get('extra_delay', 0) + 1
                still_delayed.append(packet)
                if verbose:
                    print(f"Пакет от {packet['sender_id']} для узла {recipient_id} отложен: получатель не активен (раунд {current_round})")
        else:
            still_delayed.append(packet)
    
    network_stats['delayed_packets'] = still_delayed

    packet_counter = 0
    
    for recipient_id, sender_list in messages_to_send.items():
        if not active_nodes[recipient_id]:
            for sender_id, _ in sender_list:
                dropped_agents.append({'from': sender_id, 'to': recipient_id})
            continue
        
        for sender_id, full_weights in sender_list:
            sparse_weights, mask = sparsify_weights(full_weights, sparsification_rate, base_seed, sender_id, recipient_id, current_round)
            
            packed_data = pack_sparse_message(sparse_weights, mask)
            
            # Потеря пакета
            if rng.random() < conditions.packet_loss_rate:
                lost_messages.append({'from': sender_id, 'to': recipient_id, 'round': current_round})
                continue
            
            # Задержка
            if conditions.delay_max > 0 and rng.random() < conditions.delay_probability:
                delay = rng.randint(conditions.delay_min, conditions.delay_max + 1)
                if delay > 0:
                    packet_counter += 1
                    packet_id = f"r{current_round}_s{sender_id}_r{recipient_id}_{packet_counter}"
                    
                    if 'delayed_packets' not in network_stats:
                        network_stats['delayed_packets'] = []
                    
                    network_stats['delayed_packets'].append({'packet_id': packet_id, 'recipient_id': recipient_id, 'sender_id': sender_id, 'packed_data': packed_data,
                        'delay': delay, 'original_delay': delay, 'extra_delay': 0, 'delivery_round': current_round + delay
                    })
                    delayed_messages.append({
                        'from': sender_id, 'to': recipient_id, 'delay': delay, 'reason': 'network', 
                        'packet_id': packet_id, 'source_round': current_round
                    })
                    continue
            
            # Шум
            if conditions.noise_std > 0:
                for layer_name in packed_data:
                    values = packed_data[layer_name]['values']
                    
                    if any(pattern in layer_name for pattern in ['running_mean', 'running_var', 'num_batches_tracked']):
                         packed_data[layer_name]['values'] = values
                    else:
                        noisy_values = values + np.random.randn(*values.shape).astype(np.float32) * conditions.noise_std
                        packed_data[layer_name]['values'] = noisy_values
            if recipient_id not in received:
                received[recipient_id] = []
            received[recipient_id].append({'packed_data': packed_data, 'sender_id': sender_id})
    
    stats = {
        'lost': lost_messages,
        'delayed': delayed_messages,
        'delivered_from_delay': delivered_from_delay,
        'dropped': dropped_agents,
        'received_count': len(received),
        'total_nodes': len(active_nodes),
        'round': current_round,
        'pending_delays': len(network_stats.get('delayed_packets', [])),
        'still_waiting': [p['recipient_id'] for p in network_stats.get('delayed_packets', []) if p.get('delivery_round', 0) > current_round],
        'active_nodes': active_nodes.copy()
    }
    
    return received, stats, network_stats