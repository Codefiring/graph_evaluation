#!/usr/bin/env python3
"""
图评估算法核心模块：评估驱动ioctl状态转换图的预测结果
"""

import sys
from collections import defaultdict, deque
from typing import Set, List, Tuple, Dict, Optional
import json
import csv
from pathlib import Path


class StateGraph:
    """状态转换图"""
    
    def __init__(self):
        # 状态转换表: (old_state, ioctl) -> new_state
        self.transitions: Dict[Tuple[str, str], str] = {}
        # 所有状态集合
        self.states: Set[str] = set()
        # 所有ioctl操作集合
        self.ioctls: Set[str] = set()
        # 初始状态（第一个出现的状态，或没有入边的状态）
        self.initial_state: str = None
    
    def add_transition(self, old_state: str, ioctl: str, new_state: str):
        """添加状态转换"""
        self.transitions[(old_state, ioctl)] = new_state
        self.states.add(old_state)
        self.states.add(new_state)
        self.ioctls.add(ioctl)
    
    def get_next_state(self, old_state: str, ioctl: str) -> str:
        """获取转换后的新状态"""
        return self.transitions.get((old_state, ioctl))
    
    def find_initial_state(self):
        """找到初始状态：没有入边的状态，如果多个则选择第一个出现的"""
        if self.initial_state:
            return self.initial_state
        
        # 统计每个状态的入度
        in_degree = defaultdict(int)
        for (old_state, _), new_state in self.transitions.items():
            in_degree[new_state] += 1
        
        # 找到没有入边的状态
        initial_candidates = [s for s in self.states if in_degree[s] == 0]
        
        if initial_candidates:
            # 如果有多个，选择第一个出现的（按transitions的顺序）
            for (old_state, _), _ in self.transitions.items():
                if old_state in initial_candidates:
                    self.initial_state = old_state
                    return self.initial_state
            self.initial_state = initial_candidates[0]
        else:
            # 如果没有没有入边的状态，选择第一个出现的状态
            if self.transitions:
                self.initial_state = next(iter(self.transitions))[0]
            else:
                self.initial_state = None
        
        return self.initial_state
    
    def generate_sequences(self, max_length: int, max_sequences: int = None) -> Set[Tuple[str, ...]]:
        """
        生成所有长度≤max_length的ioctl序列
        
        Args:
            max_length: 最大序列长度
            max_sequences: 最大序列数量限制（用于采样），None表示不限制
        
        Returns:
            序列集合，每个序列是一个元组 (ioctl1, ioctl2, ...)
        """
        initial = self.find_initial_state()
        if initial is None:
            return set()
        
        sequences: Set[Tuple[str, ...]] = set()
        # 空序列（长度为0）
        sequences.add(())
        
        # 使用BFS遍历所有可能的序列
        queue = deque([(initial, ())])  # (current_state, sequence)
        
        while queue:
            current_state, sequence = queue.popleft()
            
            if len(sequence) >= max_length:
                continue
            
            # 尝试所有可能的ioctl操作
            for ioctl in self.ioctls:
                next_state = self.get_next_state(current_state, ioctl)
                if next_state:
                    new_sequence = sequence + (ioctl,)
                    sequences.add(new_sequence)
                    
                    # 如果还没达到最大长度，继续扩展
                    if len(new_sequence) < max_length:
                        queue.append((next_state, new_sequence))
            
            # 如果设置了最大序列数限制，且已达到限制，停止生成
            if max_sequences and len(sequences) >= max_sequences:
                break
        
        return sequences
    
    def generate_sequences_sampled(self, max_length: int, sample_size: int = 10000) -> Set[Tuple[str, ...]]:
        """
        采样生成序列（用于分支太多的情况）
        
        Args:
            max_length: 最大序列长度
            sample_size: 采样数量
        
        Returns:
            采样的序列集合
        """
        import random
        
        initial = self.find_initial_state()
        if initial is None:
            return set()
        
        sequences: Set[Tuple[str, ...]] = set()
        sequences.add(())  # 空序列
        
        # 随机采样路径
        for _ in range(sample_size):
            current_state = initial
            sequence = []
            
            for _ in range(max_length):
                # 获取当前状态所有可能的转换
                possible_transitions = [
                    (ioctl, new_state)
                    for (old_state, ioctl), new_state in self.transitions.items()
                    if old_state == current_state
                ]
                
                if not possible_transitions:
                    break
                
                # 随机选择一个转换
                ioctl, next_state = random.choice(possible_transitions)
                sequence.append(ioctl)
                sequences.add(tuple(sequence))
                current_state = next_state
        
        return sequences


def parse_graph_file(filepath: str) -> StateGraph:
    """
    解析状态转换图文件
    
    文件格式：每行是 "old_state","ioctl","new_state" (CSV格式，字段用引号包围)
    示例：
        "NPU_VERTEX_OPEN","npu_vertex_s_graph","NPU_VERTEX_GRAPH"
    """
    graph = StateGraph()
    
    line_num = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 使用csv.reader正确解析带引号的CSV格式
            csv_reader = csv.reader(f, quotechar='"', skipinitialspace=True)
            for line_num, row in enumerate(csv_reader, 1):
                # 跳过空行
                if not row or all(not field.strip() for field in row):
                    continue
                
                # 检查字段数量
                if len(row) < 3:
                    print(f"Warning: Line {line_num} has invalid format (expected 3 fields, got {len(row)}): {row}", file=sys.stderr)
                    continue
                
                # 去除每个字段的前后空白
                old_state = row[0].strip()
                ioctl = row[1].strip()
                new_state = row[2].strip()
                
                # 检查字段是否为空
                if not old_state or not ioctl or not new_state:
                    print(f"Warning: Line {line_num} has empty field: {row}", file=sys.stderr)
                    continue
                
                graph.add_transition(old_state, ioctl, new_state)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except csv.Error as e:
        raise Exception(f"CSV parsing error in file {filepath} at line {line_num}: {e}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {e}")
    
    return graph


def evaluate_graphs(gt_file: str, pred_file: str, max_length: int, 
                    use_sampling: bool = False, sample_size: int = 10000,
                    max_sequences: int = None, verbose: bool = True) -> Dict:
    """
    评估两个状态转换图
    
    Args:
        gt_file: ground truth文件路径
        pred_file: 预测结果文件路径
        max_length: 最大序列长度k
        use_sampling: 是否使用采样（当分支太多时）
        sample_size: 采样数量
        max_sequences: 最大序列数量限制
        verbose: 是否输出详细信息
    
    Returns:
        评估指标字典，包含：
        - precision: 序列级precision
        - recall: 序列级recall
        - f1_score: F1分数
        - gt_sequences: ground truth序列数
        - pred_sequences: 预测序列数
        - common_sequences: 共同序列数
        - false_positives: 假阳性数量
        - false_negatives: 假阴性数量
        - gt_stats: ground truth统计信息
        - pred_stats: 预测结果统计信息
    """
    # 解析两个图
    if verbose:
        print(f"Parsing ground truth file: {gt_file}")
    gt_graph = parse_graph_file(gt_file)
    
    if verbose:
        print(f"Parsing prediction file: {pred_file}")
    pred_graph = parse_graph_file(pred_file)
    
    gt_stats = {
        'states': len(gt_graph.states),
        'ioctls': len(gt_graph.ioctls),
        'transitions': len(gt_graph.transitions)
    }
    
    pred_stats = {
        'states': len(pred_graph.states),
        'ioctls': len(pred_graph.ioctls),
        'transitions': len(pred_graph.transitions)
    }
    
    if verbose:
        print(f"Ground truth: {gt_stats['states']} states, {gt_stats['ioctls']} ioctls, {gt_stats['transitions']} transitions")
        print(f"Prediction: {pred_stats['states']} states, {pred_stats['ioctls']} ioctls, {pred_stats['transitions']} transitions")
    
    # 生成序列
    if verbose:
        print(f"\nGenerating sequences (max_length={max_length})...")
    
    if use_sampling:
        if verbose:
            print("Using sampling method...")
        gt_sequences = gt_graph.generate_sequences_sampled(max_length, sample_size)
        pred_sequences = pred_graph.generate_sequences_sampled(max_length, sample_size)
    else:
        gt_sequences = gt_graph.generate_sequences(max_length, max_sequences)
        pred_sequences = pred_graph.generate_sequences(max_length, max_sequences)
    
    if verbose:
        print(f"Ground truth sequences: {len(gt_sequences)}")
        print(f"Prediction sequences: {len(pred_sequences)}")
    
    # 计算交集
    intersection = gt_sequences & pred_sequences
    if verbose:
        print(f"Common sequences: {len(intersection)}")
    
    # 计算precision和recall
    if len(pred_sequences) > 0:
        precision = len(intersection) / len(pred_sequences)
    else:
        precision = 0.0
    
    if len(gt_sequences) > 0:
        recall = len(intersection) / len(gt_sequences)
    else:
        recall = 0.0
    
    # 计算F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # 计算其他指标
    false_positives = pred_sequences - gt_sequences
    false_negatives = gt_sequences - pred_sequences
    
    results = {
        'metric': 'sequence',
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gt_sequences': len(gt_sequences),
        'pred_sequences': len(pred_sequences),
        'common_sequences': len(intersection),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'gt_stats': gt_stats,
        'pred_stats': pred_stats,
    }
    
    return results


def find_topology_mapping(gt_graph: StateGraph, pred_graph: StateGraph) -> Dict[str, str]:
    """
    找到两个图之间的状态映射，基于拓扑结构相似性
    
    使用贪心算法，基于以下特征匹配状态：
    1. 节点的度（入度+出度）
    2. 连接的ioctl操作集合
    3. 连接的ioctl操作的模式（作为输入/输出的ioctl）
    
    Returns:
        从pred_graph状态名到gt_graph状态名的映射字典
    """
    from collections import defaultdict
    
    # 计算每个状态的拓扑特征
    def compute_node_features(graph: StateGraph, state: str) -> Tuple[int, int, Set[str], Set[str]]:
        """返回 (入度, 出度, 输入ioctl集合, 输出ioctl集合)"""
        in_ioctls = set()
        out_ioctls = set()
        in_degree = 0
        out_degree = 0
        
        for (old_state, ioctl), new_state in graph.transitions.items():
            if old_state == state:
                out_ioctls.add(ioctl)
                out_degree += 1
            if new_state == state:
                in_ioctls.add(ioctl)
                in_degree += 1
        
        return (in_degree, out_degree, in_ioctls, out_ioctls)
    
    # 为每个状态计算特征
    gt_features = {}
    for state in gt_graph.states:
        gt_features[state] = compute_node_features(gt_graph, state)
    
    pred_features = {}
    for state in pred_graph.states:
        pred_features[state] = compute_node_features(pred_graph, state)
    
    # 贪心匹配：为每个pred状态找到最相似的gt状态
    mapping = {}
    used_gt_states = set()
    
    # 按特征相似度排序候选匹配
    for pred_state, pred_feat in pred_features.items():
        best_match = None
        best_score = -1
        
        for gt_state, gt_feat in gt_features.items():
            if gt_state in used_gt_states:
                continue
            
            # 计算相似度分数
            # 1. 度匹配（权重0.3）
            degree_score = 0.0
            if pred_feat[0] == gt_feat[0] and pred_feat[1] == gt_feat[1]:
                degree_score = 1.0
            elif abs(pred_feat[0] - gt_feat[0]) + abs(pred_feat[1] - gt_feat[1]) <= 1:
                degree_score = 0.5
            
            # 2. ioctl集合匹配（权重0.7）
            in_ioctl_match = len(pred_feat[2] & gt_feat[2]) / max(len(pred_feat[2] | gt_feat[2]), 1)
            out_ioctl_match = len(pred_feat[3] & gt_feat[3]) / max(len(pred_feat[3] | gt_feat[3]), 1)
            ioctl_score = (in_ioctl_match + out_ioctl_match) / 2
            
            # 综合分数
            score = 0.3 * degree_score + 0.7 * ioctl_score
            
            if score > best_score:
                best_score = score
                best_match = gt_state
        
        if best_match and best_score > 0.1:  # 阈值，避免错误匹配
            mapping[pred_state] = best_match
            used_gt_states.add(best_match)
    
    return mapping


def compute_topology_based_edit_distance(gt_graph: StateGraph, pred_graph: StateGraph,
                                         node_ins_cost: float = 1.0, node_del_cost: float = 1.0,
                                         edge_ins_cost: float = 1.0, edge_del_cost: float = 1.0) -> Dict:
    """
    基于拓扑结构的图编辑距离计算
    
    首先找到状态映射，然后基于映射后的状态名称比较图结构。
    如果无法找到映射，则基于ioctl操作的模式来比较。
    """
    # 找到状态映射
    state_mapping = find_topology_mapping(gt_graph, pred_graph)
    
    # 创建映射后的pred图（使用gt的状态名称）
    mapped_pred_edges = set()
    for (old_state, ioctl), new_state in pred_graph.transitions.items():
        mapped_old = state_mapping.get(old_state, old_state)
        mapped_new = state_mapping.get(new_state, new_state)
        mapped_pred_edges.add((mapped_old, ioctl, mapped_new))
    
    # GT图的边集合
    gt_edges = set((old_state, ioctl, new_state)
                   for (old_state, ioctl), new_state in gt_graph.transitions.items())
    
    # 计算编辑距离
    edge_deletions = gt_edges - mapped_pred_edges
    edge_insertions = mapped_pred_edges - gt_edges
    
    # 节点匹配情况
    mapped_pred_nodes = set(state_mapping.values())
    gt_nodes = set(gt_graph.states)
    pred_nodes = set(pred_graph.states)
    
    node_deletions = gt_nodes - mapped_pred_nodes
    node_insertions = set(pred_nodes) - set(state_mapping.keys())
    
    edit_cost = (
        node_del_cost * len(node_deletions) +
        node_ins_cost * len(node_insertions) +
        edge_del_cost * len(edge_deletions) +
        edge_ins_cost * len(edge_insertions)
    )
    
    denom = (
        node_del_cost * len(gt_nodes) +
        node_ins_cost * len(pred_nodes) +
        edge_del_cost * len(gt_edges) +
        edge_ins_cost * len(mapped_pred_edges)
    )
    
    normalized_distance = (edit_cost / denom) if denom > 0 else 0.0
    similarity = 1.0 - normalized_distance
    
    return {
        'edit_distance': edit_cost,
        'normalized_distance': normalized_distance,
        'similarity': similarity,
        'node_deletions': len(node_deletions),
        'node_insertions': len(node_insertions),
        'edge_deletions': len(edge_deletions),
        'edge_insertions': len(edge_insertions),
        'state_mapping': state_mapping,
        'mapping_coverage': len(state_mapping) / max(len(pred_nodes), 1),
        'gt_stats': {
            'states': len(gt_nodes),
            'ioctls': len(gt_graph.ioctls),
            'transitions': len(gt_edges)
        },
        'pred_stats': {
            'states': len(pred_nodes),
            'ioctls': len(pred_graph.ioctls),
            'transitions': len(pred_graph.transitions)
        }
    }


def compute_graph_edit_distance(gt_graph: StateGraph, pred_graph: StateGraph,
                                node_ins_cost: float = 1.0, node_del_cost: float = 1.0,
                                edge_ins_cost: float = 1.0, edge_del_cost: float = 1.0) -> Dict:
    """
    Compute a simple graph edit distance based on exact node/edge label matching.

    Nodes are matched by name; edges are matched by (old_state, ioctl, new_state).
    """
    gt_nodes = set(gt_graph.states)
    pred_nodes = set(pred_graph.states)

    gt_edges = set((old_state, ioctl, new_state)
                   for (old_state, ioctl), new_state in gt_graph.transitions.items())
    pred_edges = set((old_state, ioctl, new_state)
                     for (old_state, ioctl), new_state in pred_graph.transitions.items())

    node_deletions = gt_nodes - pred_nodes
    node_insertions = pred_nodes - gt_nodes
    edge_deletions = gt_edges - pred_edges
    edge_insertions = pred_edges - gt_edges

    edit_cost = (
        node_del_cost * len(node_deletions) +
        node_ins_cost * len(node_insertions) +
        edge_del_cost * len(edge_deletions) +
        edge_ins_cost * len(edge_insertions)
    )

    denom = (
        node_del_cost * len(gt_nodes) +
        node_ins_cost * len(pred_nodes) +
        edge_del_cost * len(gt_edges) +
        edge_ins_cost * len(pred_edges)
    )

    normalized_distance = (edit_cost / denom) if denom > 0 else 0.0
    similarity = 1.0 - normalized_distance

    return {
        'edit_distance': edit_cost,
        'normalized_distance': normalized_distance,
        'similarity': similarity,
        'node_deletions': len(node_deletions),
        'node_insertions': len(node_insertions),
        'edge_deletions': len(edge_deletions),
        'edge_insertions': len(edge_insertions),
        'gt_stats': {
            'states': len(gt_nodes),
            'ioctls': len(gt_graph.ioctls),
            'transitions': len(gt_edges)
        },
        'pred_stats': {
            'states': len(pred_nodes),
            'ioctls': len(pred_graph.ioctls),
            'transitions': len(pred_edges)
        }
    }


def evaluate_graphs_edit_distance(gt_file: str, pred_file: str, verbose: bool = True,
                                  node_ins_cost: float = 1.0, node_del_cost: float = 1.0,
                                  edge_ins_cost: float = 1.0, edge_del_cost: float = 1.0) -> Dict:
    """
    Evaluate two graphs using a simple graph edit distance.
    """
    if verbose:
        print(f"Parsing ground truth file: {gt_file}")
    gt_graph = parse_graph_file(gt_file)

    if verbose:
        print(f"Parsing prediction file: {pred_file}")
    pred_graph = parse_graph_file(pred_file)

    results = compute_graph_edit_distance(
        gt_graph,
        pred_graph,
        node_ins_cost=node_ins_cost,
        node_del_cost=node_del_cost,
        edge_ins_cost=edge_ins_cost,
        edge_del_cost=edge_del_cost
    )
    results['metric'] = 'edit_distance'
    return results


def evaluate_graphs_topology(gt_file: str, pred_file: str, max_length: int = 5,
                             use_sampling: bool = False, sample_size: int = 10000,
                             max_sequences: int = None, verbose: bool = True,
                             node_ins_cost: float = 1.0, node_del_cost: float = 1.0,
                             edge_ins_cost: float = 1.0, edge_del_cost: float = 1.0) -> Dict:
    """
    基于拓扑结构的图评估算法
    
    评估两个状态转换图，即使状态名称不同，只要拓扑结构相同就能正确评估。
    结合了序列评估和拓扑编辑距离两种方法。
    
    Args:
        gt_file: ground truth文件路径
        pred_file: 预测结果文件路径
        max_length: 最大序列长度k
        use_sampling: 是否使用采样
        sample_size: 采样数量
        max_sequences: 最大序列数量限制
        verbose: 是否输出详细信息
        node_ins_cost, node_del_cost, edge_ins_cost, edge_del_cost: 编辑距离成本
    
    Returns:
        评估指标字典
    """
    if verbose:
        print(f"Parsing ground truth file: {gt_file}")
    gt_graph = parse_graph_file(gt_file)
    
    if verbose:
        print(f"Parsing prediction file: {pred_file}")
    pred_graph = parse_graph_file(pred_file)
    
    # 找到状态映射
    if verbose:
        print("\nFinding topology-based state mapping...")
    state_mapping = find_topology_mapping(gt_graph, pred_graph)
    
    if verbose:
        print(f"Found {len(state_mapping)} state mappings:")
        for pred_state, gt_state in state_mapping.items():
            print(f"  {pred_state} -> {gt_state}")
    
    # 创建映射后的pred图（用于序列比较）
    mapped_pred_graph = StateGraph()
    for (old_state, ioctl), new_state in pred_graph.transitions.items():
        mapped_old = state_mapping.get(old_state, old_state)
        mapped_new = state_mapping.get(new_state, new_state)
        mapped_pred_graph.add_transition(mapped_old, ioctl, mapped_new)
    
    # 基于拓扑的编辑距离
    edit_results = compute_topology_based_edit_distance(
        gt_graph, pred_graph,
        node_ins_cost=node_ins_cost,
        node_del_cost=node_del_cost,
        edge_ins_cost=edge_ins_cost,
        edge_del_cost=edge_del_cost
    )
    
    # 基于映射后的序列评估
    if verbose:
        print(f"\nGenerating sequences (max_length={max_length})...")
    
    if use_sampling:
        if verbose:
            print("Using sampling method...")
        gt_sequences = gt_graph.generate_sequences_sampled(max_length, sample_size)
        pred_sequences = mapped_pred_graph.generate_sequences_sampled(max_length, sample_size)
    else:
        gt_sequences = gt_graph.generate_sequences(max_length, max_sequences)
        pred_sequences = mapped_pred_graph.generate_sequences(max_length, max_sequences)
    
    if verbose:
        print(f"Ground truth sequences: {len(gt_sequences)}")
        print(f"Prediction sequences (mapped): {len(pred_sequences)}")
    
    # 计算交集
    intersection = gt_sequences & pred_sequences
    if verbose:
        print(f"Common sequences: {len(intersection)}")
    
    # 计算precision和recall
    if len(pred_sequences) > 0:
        precision = len(intersection) / len(pred_sequences)
    else:
        precision = 0.0
    
    if len(gt_sequences) > 0:
        recall = len(intersection) / len(gt_sequences)
    else:
        recall = 0.0
    
    # 计算F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # 计算其他指标
    false_positives = pred_sequences - gt_sequences
    false_negatives = gt_sequences - pred_sequences
    
    results = {
        'metric': 'topology',
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gt_sequences': len(gt_sequences),
        'pred_sequences': len(pred_sequences),
        'common_sequences': len(intersection),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'edit_distance': edit_results['edit_distance'],
        'normalized_distance': edit_results['normalized_distance'],
        'similarity': edit_results['similarity'],
        'state_mapping': state_mapping,
        'mapping_coverage': edit_results['mapping_coverage'],
        'gt_stats': {
            'states': len(gt_graph.states),
            'ioctls': len(gt_graph.ioctls),
            'transitions': len(gt_graph.transitions)
        },
        'pred_stats': {
            'states': len(pred_graph.states),
            'ioctls': len(pred_graph.ioctls),
            'transitions': len(pred_graph.transitions)
        }
    }
    
    return results


def save_results_json(results, output_file: str):
    """
    将评估结果保存为JSON格式
    
    Args:
        results: 可以是单个结果字典(Dict)或结果列表(List[Dict])
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_results_csv(results_list: List[Dict], output_file: str):
    """
    将多个驱动的评估结果保存为CSV格式
    
    Args:
        results_list: 结果列表，每个元素包含 'driver_name' 和评估指标
        output_file: 输出文件路径
    """
    if not results_list:
        return
    
    # 获取所有字段名
    fieldnames = ['driver_name']
    # 从第一个结果中提取所有指标字段（排除driver_name）
    for key in results_list[0].keys():
        if key != 'driver_name':
            if isinstance(results_list[0][key], dict):
                # 如果是嵌套字典，展开
                for sub_key in results_list[0][key].keys():
                    fieldnames.append(f"{key}_{sub_key}")
            else:
                fieldnames.append(key)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_list:
            row = {'driver_name': result.get('driver_name', '')}
            for key, value in result.items():
                if key == 'driver_name':
                    continue
                if isinstance(value, dict):
                    # 展开嵌套字典
                    for sub_key, sub_value in value.items():
                        row[f"{key}_{sub_key}"] = sub_value
                else:
                    row[key] = value
            writer.writerow(row)
