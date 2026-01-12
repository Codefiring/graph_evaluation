#!/usr/bin/env python3
"""
图评估算法：评估驱动ioctl状态转换图的预测结果

输入：
- ground truth文件
- 预测结果文件
- 最大序列长度k

输出：
- 序列级precision
- 序列级recall
- 其他评估指标
"""

import sys
from collections import defaultdict, deque
from typing import Set, List, Tuple, Dict
import argparse


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
    
    文件格式：每行是 "old_state,ioctl,new_state"
    """
    graph = StateGraph()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 解析CSV格式，处理引号
                parts = [p.strip().strip('"') for p in line.split(',')]
                if len(parts) < 3:
                    print(f"Warning: Line {line_num} has invalid format: {line}", file=sys.stderr)
                    continue
                
                old_state = parts[0]
                ioctl = parts[1]
                new_state = parts[2]
                
                graph.add_transition(old_state, ioctl, new_state)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return graph


def evaluate_graphs(gt_file: str, pred_file: str, max_length: int, 
                    use_sampling: bool = False, sample_size: int = 10000,
                    max_sequences: int = None) -> Dict[str, float]:
    """
    评估两个状态转换图
    
    Args:
        gt_file: ground truth文件路径
        pred_file: 预测结果文件路径
        max_length: 最大序列长度k
        use_sampling: 是否使用采样（当分支太多时）
        sample_size: 采样数量
        max_sequences: 最大序列数量限制
    
    Returns:
        评估指标字典
    """
    # 解析两个图
    print(f"Parsing ground truth file: {gt_file}")
    gt_graph = parse_graph_file(gt_file)
    
    print(f"Parsing prediction file: {pred_file}")
    pred_graph = parse_graph_file(pred_file)
    
    print(f"Ground truth: {len(gt_graph.states)} states, {len(gt_graph.ioctls)} ioctls, {len(gt_graph.transitions)} transitions")
    print(f"Prediction: {len(pred_graph.states)} states, {len(pred_graph.ioctls)} ioctls, {len(pred_graph.transitions)} transitions")
    
    # 生成序列
    print(f"\nGenerating sequences (max_length={max_length})...")
    
    if use_sampling:
        print("Using sampling method...")
        gt_sequences = gt_graph.generate_sequences_sampled(max_length, sample_size)
        pred_sequences = pred_graph.generate_sequences_sampled(max_length, sample_size)
    else:
        gt_sequences = gt_graph.generate_sequences(max_length, max_sequences)
        pred_sequences = pred_graph.generate_sequences(max_length, max_sequences)
    
    print(f"Ground truth sequences: {len(gt_sequences)}")
    print(f"Prediction sequences: {len(pred_sequences)}")
    
    # 计算交集
    intersection = gt_sequences & pred_sequences
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
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='评估驱动ioctl状态转换图的预测结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python graph_evaluator.py gt.txt pred.txt --max-length 5
  python graph_evaluator.py gt.txt pred.txt --max-length 6 --sampling --sample-size 50000
        """
    )
    
    parser.add_argument('gt_file', help='Ground truth文件路径')
    parser.add_argument('pred_file', help='预测结果文件路径')
    parser.add_argument('--max-length', '-k', type=int, default=5,
                       help='最大序列长度k (默认: 5)')
    parser.add_argument('--sampling', action='store_true',
                       help='使用采样方法（当分支太多时）')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='采样数量 (默认: 10000)')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='最大序列数量限制（用于控制内存）')
    
    args = parser.parse_args()
    
    # 执行评估
    results = evaluate_graphs(
        args.gt_file,
        args.pred_file,
        args.max_length,
        use_sampling=args.sampling,
        sample_size=args.sample_size,
        max_sequences=args.max_sequences
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"序列级 Precision: {results['precision']:.4f}")
    print(f"序列级 Recall:     {results['recall']:.4f}")
    print(f"F1 Score:          {results['f1_score']:.4f}")
    print(f"\n详细统计:")
    print(f"  Ground truth 序列数:  {results['gt_sequences']}")
    print(f"  预测结果序列数:       {results['pred_sequences']}")
    print(f"  共同序列数:           {results['common_sequences']}")
    print(f"  假阳性 (False Pos):   {results['false_positives']}")
    print(f"  假阴性 (False Neg):   {results['false_negatives']}")
    print("="*60)


if __name__ == '__main__':
    main()
