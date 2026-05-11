import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import torch
import math
from split_data import split_data

colors = [
    '#FF6B6B', '#45B7D1', '#96CEB4', '#3498DB', '#4ECDC4', '#2ECC71', '#FF8E53', '#9B59B6', '#FFB347', '#F1C40F', '#E67E22', 
    '#1ABC9C', '#E74C3C', '#F39C12', '#16A085', '#C0392B', '#2980B9', '#8E44AD', '#D35400', '#27AE60', '#FF4757', '#70A1FF', 
    '#7BED9F', '#FFA502', '#A4B0BE', '#2ED573', '#FF7F50', '#6C5CE7', '#00CEC9', '#FF9FF3', '#FECA57', '#48DBFB', '#E1B12C', 
    '#FDA7DF', '#D980FA', '#B8E994', '#FAB1A0', '#55E6C1', '#81ECE5', '#FFC312', '#C4E538', '#12CBC4', '#FDAE8A', '#D6A2E8'
]

def visualize_agents_distribution(train_dataset, num_agents, distribution='dirichlet', alpha=0.5, random_seed=42, colors=None, title_prefix=""):
    """
    Визуализирует распределение классов по агентам после разбиения датасета
    """
    agent_datasets = split_data(train_dataset, num_agents, distribution, alpha, random_seed, verbose=False)

    all_labels = []
    for i in range(len(train_dataset)):
        if isinstance(train_dataset[i], tuple) and len(train_dataset[i]) == 2:
            _, label = train_dataset[i]
        elif isinstance(train_dataset[i], dict):
            label = train_dataset[i]['label']
        else:
            continue
        all_labels.append(label)
    num_classes = len(set(all_labels))
    class_names = [str(c) for c in sorted(set(all_labels))]

    agent_class_counts = []
    for subset in agent_datasets:
        labels = [train_dataset[idx][1] for idx in subset.indices]
        counts = Counter(labels)
        class_dist = [counts.get(c, 0) for c in range(num_classes)]
        agent_class_counts.append(class_dist)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bottom = np.zeros(num_agents)
    for class_id in range(num_classes):
        counts = [agent_class_counts[agent][class_id] for agent in range(num_agents)]
        ax.bar(range(num_agents), counts, bottom=bottom, 
               label=f'{class_names[class_id]}', color=colors[class_id], edgecolor='white', linewidth=0.8)
        bottom += counts

    ax.set_xlabel('Агент', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество примеров', fontsize=12, fontweight='bold')
    if distribution == 'dirichlet':
        dist_name = f'Dirichlet (α={alpha})'
    else:
        dist_name = 'IID'
    ax.set_title(f'{title_prefix}Распределение классов между {num_agents} агентами: {dist_name}', fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8, color='#aaaaaa')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title='Класс', fontsize=9)
    ax.set_xticks(range(num_agents))
    ax.set_xticklabels([str(i) for i in range(num_agents)], fontsize=10)
    for i, total in enumerate(bottom):
        ax.text(i, total + (max(bottom) * 0.01), f'{int(total)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

def visualize_dataset_classes(colors, train_dataset, test_dataset, mean_val, std_val):
    """
    Визуализирует классы датасета и их распределение
    """
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
        unique_labels = set(range(len(class_names)))
        classes = class_names
        use_real_names = True
    else:
        unique_labels = set()
        for item in train_dataset:
            if isinstance(item, dict):
                label = item['label']
            elif isinstance(item, tuple) and len(item) == 2:
                _, label = item
            else:
                continue
            unique_labels.add(label)
        classes = [str(i) for i in sorted(unique_labels)]
        class_names = classes
        use_real_names = False
    
    num_classes = len(classes)
    print(f"Найдено классов: {num_classes}")
    print(f"Названия классов: {classes}")
    
    train_counts = Counter()
    for item in train_dataset:
        if isinstance(item, dict):
            label = item['label']
        else:
            _, label = item
        train_counts[label] += 1
    
    test_counts = Counter()
    for item in test_dataset:
        if isinstance(item, dict):
            label = item['label']
        else:
            _, label = item
        test_counts[label] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    if use_real_names:
        train_values = [train_counts[i] for i in range(num_classes)]
        test_values = [test_counts[i] for i in range(num_classes)]
    else:
        train_values = [train_counts[int(cls)] for cls in classes]
        test_values = [test_counts[int(cls)] for cls in classes]
    
    total_train = len(train_dataset)
    total_test = len(test_dataset)
    y_pos = np.arange(num_classes)
    
    setup_bar_plot(axes[0], y_pos, classes, train_values, total_train, f'Распределение классов в тренировочном датасете\n(всего: {total_train} изображений)', colors[0])
    
    setup_bar_plot(axes[1], y_pos, classes, test_values, total_test, f'Распределение классов в тестовом датасете\n(всего: {total_test} изображений)', colors[1])
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, num_classes, figsize=(min(4 * num_classes, 20), 4))
    fig.patch.set_facecolor('white')
    
    if num_classes == 1:
        axes = [axes]
    
    shown = {}
    for item in train_dataset:
        if isinstance(item, dict):
            img = item['image']
            label = item['label']
        else:
            img, label = item
            
        if label not in shown:
            shown[label] = img
        if len(shown) == num_classes:
            break
    
    for i, (label, ax) in enumerate(zip(sorted(shown.keys()), axes)):
        img = shown[label]
        
        if torch.is_tensor(img):
            if img.shape[0] == 1:
                img = img.squeeze(0)
            else:
                img = img.permute(1, 2, 0)
            img = img * std_val + mean_val
            img = torch.clamp(img, 0, 1)
            img = img.numpy()
        else:
            img = np.array(img) / 255.0
        
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze()
        
        ax.imshow(img, cmap='gray')
        if use_real_names:
            title_text = class_names[label]
        else:
            title_text = str(label)

        ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    train_values_array = np.array(train_values)
    imbalance_ratio = max(train_values_array) / min(train_values_array)
    print(f"\nСоотношение (макс/мин): {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        print("Классы не сбалансированы!")
    else:
        print("Классы сбалансированы!")

def setup_bar_plot(ax, y_pos, classes, values, total, title, color):
    ax.set_facecolor('white')
    bars = ax.barh(y_pos, values, color=color, edgecolor='white', linewidth=1, alpha=0.8, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Количество изображений', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8, color='#aaaaaa')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    
    max_x = max(values)
    for bar, value in zip(bars, values):
        width = bar.get_width()
        percentage = (value / total) * 100
        ax.text(width + (max_x * 0.01), bar.get_y() + bar.get_height()/2,
                f'{value} ({percentage:.1f}%)', ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', bottom=True, left=True, length=4, width=1, color='black')
    ax.set_xlim(0, max_x * 1.15)
    return ax