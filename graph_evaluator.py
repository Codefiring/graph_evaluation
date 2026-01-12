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


def save_results_json(results: Dict, output_file: str):
    """将评估结果保存为JSON格式"""
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
