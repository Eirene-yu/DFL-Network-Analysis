import warnings
import torch
import torch.nn as nn
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix

def init_metrics_dict(return_all_metrics):
    """
    Инициализация словаря метрик
    """
    if not return_all_metrics:
        return {
            'accuracy': [], 'loss': [], 'consensus_error': [],
            'lost': [], 'delayed': [], 'dropped': [],
            'accuracy_active': [], 'loss_active': [], 'consensus_error_active': [],
            'inactive': [],
            'agent': {
                'accuracy': [],
                'loss': []
            }
        }
    
    metric_dic = {
        'accuracy': [], 'loss': [], 'consensus_error': [], 'final': {},
        'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
        'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [],
        'confidence_mean': [], 'confidence_std': [],
        'accuracy_active': [], 'loss_active': [], 'consensus_error_active': [],
        'inactive': [],
        'agent': {
            'accuracy': [],
            'loss': []
        }
        , 'lost': [], 'delayed': [], 'dropped': []
    }
    
    return metric_dic

def log_current_metrics(agents, metric_dic, test_loader, model_class, agent_weights, return_all_metrics, network_stats=None, is_final_round=False, return_agent_metrics=False, active_nodes=None):
    """
    Записывает текущее состояние системы в метрики
    """
    all_weights = [agent.get_weights() for agent in agents]

    active_indices = None
    num_active = 0
    num_total = 0
    
    if active_nodes is not None:
        active_indices = [i for i, is_active in enumerate(active_nodes) if is_active]
        num_active = len(active_indices)
        num_total = len(active_nodes)

        inactive_indices = [i for i, is_active in enumerate(active_nodes) if not is_active]
        metric_dic['inactive'].append(inactive_indices)
    
    # Consensus error
    consensus_error = compute_consensus_error(all_weights)
    metric_dic['consensus_error'].append(consensus_error)

    if active_nodes is None:
        consensus_error_active = consensus_error
    else:
        if num_active == num_total:
            consensus_error_active = consensus_error
        elif num_active > 0:
            active_weights = [all_weights[i] for i in active_indices]
            consensus_error_active = compute_consensus_error(active_weights)
        else:
            consensus_error_active = -1.0
    
    metric_dic['consensus_error_active'].append(consensus_error_active)
    
    # Сетевые метрики
    if network_stats:
        metric_dic['lost'].append(network_stats.get('lost', []))
        metric_dic['delayed'].append(network_stats.get('delayed', []))
        metric_dic['dropped'].append(network_stats.get('dropped', []))
    else:
        metric_dic['lost'].append([])
        metric_dic['delayed'].append([])
        metric_dic['dropped'].append([])
    
    # Оценка каждого агента
    if (return_agent_metrics or is_final_round) and test_loader is not None and model_class is not None:
        agent_accuracies = []
        agent_losses = []
        for agent in agents:
            temp_model = model_class()
            temp_model.load_state_dict(agent.get_weights())
            agent_acc, agent_loss = evaluate_agent(temp_model, test_loader)
            agent_accuracies.append(agent_acc)
            agent_losses.append(agent_loss)
        metric_dic['agent']['accuracy'].append(agent_accuracies)
        metric_dic['agent']['loss'].append(agent_losses)
    
    # Глобальная оценка
    if test_loader is not None and model_class is not None:
        avg_weights = average_weights(all_weights)
    
        if avg_weights is not None:
            temp_model = model_class()
            temp_model.load_state_dict(avg_weights)
            eval_result = evaluate_light(temp_model, test_loader, return_all_metrics)
            metric_dic['accuracy'].append(eval_result['accuracy'])
            metric_dic['loss'].append(eval_result['loss'])

            active_acc = eval_result['accuracy']
            active_loss = eval_result['loss']

            # Оценка активных агентов
            if active_nodes is not None:
                if num_active < num_total and num_active > 0:
                    active_weights = [all_weights[i] for i in active_indices]
                    avg_active_weights = average_weights(active_weights)

                    if avg_active_weights is not None:
                        temp_model_active = model_class()
                        temp_model_active.load_state_dict(avg_active_weights)
                        eval_result_active = evaluate_light(temp_model_active, test_loader, return_all_metrics)
                        
                        active_acc = eval_result_active['accuracy']
                        active_loss = eval_result_active['loss']
                    else:
                        active_acc = 0.0
                        active_loss = 0.0
                elif num_active == 0:
                    active_acc = 0.0
                    active_loss = 0.0

            metric_dic['accuracy_active'].append(active_acc)
            metric_dic['loss_active'].append(active_loss)
            
            if return_all_metrics:
                for key, value in eval_result.items():
                    if key not in ['accuracy', 'loss'] and key in metric_dic:
                        metric_dic[key].append(value)
            if is_final_round and return_all_metrics:
                eval_result_final = evaluate_final(temp_model, test_loader)
                metric_dic['final'] = eval_result_final
        else:
            metric_dic['accuracy'].append(0.0)
            metric_dic['loss'].append(0.0)
            metric_dic['accuracy_active'].append(0.0)
            metric_dic['loss_active'].append(0.0)
    
    return metric_dic
    

def evaluate_agent(model, test_loader):
    """
    Оценка отдельного агента на тестовых данных.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            loss = loss_fn(output, y)
            pred = output.argmax(dim=1)
            
            total += y.size(0)
            correct += (pred == y).sum().item()
            total_loss += loss.item() * y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss

def evaluate_light(model, test_loader, return_all_metrics=False):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    model.eval()
    
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)

            loss = loss_fn(output, y)
            total_loss += loss.item() * x.size(0)

            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += x.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / total
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss
    }
    
    if not return_all_metrics:
        return metrics
    
    # Macro и Weighted average
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Статистика уверенности
    max_probs = np.max(all_probs, axis=1)
    confidence_mean = np.mean(max_probs) if len(max_probs) > 0 else 0.0
    confidence_std = np.std(max_probs) if len(max_probs) > 0 else 0.0
    
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confidence_mean': confidence_mean,
        'confidence_std': confidence_std,
    })
    
    return metrics

def evaluate_final(model, test_loader, light_metrics=None):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    total = len(all_labels)

    if total == 0:
        return {
            'precision_per_class': [], 'recall_per_class': [], 'f1_per_class': [],
            'roc_auc_ovr': {}, 'roc_auc_macro': 0.0, 'class_support': [],
            'confusion_matrix': [], 'classification_report': {},
            'predictions': [], 'true_labels': [], 'probabilities': [],
            'test_samples': 0, 'n_classes': 0,
        }
    
    if np.any(np.isnan(all_probs)):
        all_probs = np.nan_to_num(all_probs, nan=0.0)

    n_classes = len(np.unique(all_labels))
    
    # Per-class метрики
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # ROC AUC
    roc_auc_ovr = {}
    roc_auc_macro = 0.0
    
    if len(all_probs.shape) == 2 and all_probs.shape[1] > 1:
        if not np.any(np.isnan(all_probs)) and np.all(np.isfinite(all_probs)):
            unique_labels = np.unique(all_labels)
            for i in unique_labels:
                y_binary = (all_labels == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # есть оба класса
                    roc_auc_ovr[f'class_{i}'] = roc_auc_score(y_binary, all_probs[:, i])
                else:
                    roc_auc_ovr[f'class_{i}'] = None
            
            # Macro ROC AUC
            if len(unique_labels) == all_probs.shape[1]:
                roc_auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            else:
                roc_auc_macro = 0.0
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    class_support = [report[str(i)]['support'] for i in range(n_classes)] if n_classes > 0 else []
    
    metrics = {
        'precision_per_class': precision_per_class.tolist() if isinstance(precision_per_class, np.ndarray) else [],
        'recall_per_class': recall_per_class.tolist() if isinstance(recall_per_class, np.ndarray) else [],
        'f1_per_class': f1_per_class.tolist() if isinstance(f1_per_class, np.ndarray) else [],
        'roc_auc_ovr': roc_auc_ovr,
        'roc_auc_macro': roc_auc_macro,
        'class_support': class_support,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'predictions': all_preds.tolist(),
        'true_labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'test_samples': total,
        'n_classes': n_classes,
    }
    
    return metrics

def compute_consensus_error(weights_list):
    """
    Средняя ошибка консенсуса между агентами
    """
    if not weights_list:
        return 0.0

    avg_weights = average_weights(weights_list)
    total_diff = 0.0

    for w in weights_list:
        diff = 0.0
        for key in w.keys():
            diff += torch.norm(w[key] - avg_weights[key]) ** 2
        total_diff += diff.item()

    return total_diff / len(weights_list)

def average_weights(weights_list):
    """
    Усреднение весов
    """
    if not weights_list:
        return None
    
    avg = {}
    for key in weights_list[0].keys():
        avg[key] = sum([w[key] for w in weights_list]) / len(weights_list)
    return avg