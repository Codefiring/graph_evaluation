#!/usr/bin/env python3
"""
单文件评估程序：保持向后兼容，支持单文件评估
"""

import argparse
import sys
from graph_evaluator import evaluate_graphs


def main():
    parser = argparse.ArgumentParser(
        description='评估驱动ioctl状态转换图的预测结果（单文件模式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate_single.py gt.txt pred.txt --max-length 5
  python evaluate_single.py gt.txt pred.txt --max-length 6 --sampling --sample-size 50000
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
        max_sequences=args.max_sequences,
        verbose=True
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
