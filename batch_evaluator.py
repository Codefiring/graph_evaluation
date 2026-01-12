#!/usr/bin/env python3
"""
批量评估程序：从配置文件读取多个驱动的评估任务，批量执行并保存结果
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from graph_evaluator import evaluate_graphs, save_results_json, save_results_csv


def load_config(config_file: str) -> Dict:
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        sys.exit(1)


def validate_config(config: Dict) -> bool:
    """验证配置文件格式"""
    if 'drivers' not in config:
        print("Error: Config file must contain 'drivers' field", file=sys.stderr)
        return False
    
    if not isinstance(config['drivers'], list):
        print("Error: 'drivers' must be a list", file=sys.stderr)
        return False
    
    for i, driver in enumerate(config['drivers']):
        if not isinstance(driver, dict):
            print(f"Error: Driver {i} must be a dictionary", file=sys.stderr)
            return False
        
        if 'name' not in driver:
            print(f"Error: Driver {i} must have 'name' field", file=sys.stderr)
            return False
        
        if 'gt_file' not in driver:
            print(f"Error: Driver {i} must have 'gt_file' field", file=sys.stderr)
            return False
        
        if 'pred_file' not in driver:
            print(f"Error: Driver {i} must have 'pred_file' field", file=sys.stderr)
            return False
    
    return True


def batch_evaluate(config_file: str, output_dir: str = "results", 
                   output_format: str = "both", verbose: bool = True):
    """
    批量评估多个驱动
    
    Args:
        config_file: 配置文件路径
        output_dir: 输出目录
        output_format: 输出格式，'json', 'csv', 或 'both'
        verbose: 是否输出详细信息
    """
    # 加载配置
    config = load_config(config_file)
    
    # 验证配置
    if not validate_config(config):
        sys.exit(1)
    
    # 获取全局参数（如果存在）
    global_params = config.get('parameters', {})
    max_length = global_params.get('max_length', 5)
    use_sampling = global_params.get('use_sampling', False)
    sample_size = global_params.get('sample_size', 10000)
    max_sequences = global_params.get('max_sequences', None)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = []
    summary_results = []
    
    # 遍历每个驱动
    drivers = config['drivers']
    total = len(drivers)
    
    if verbose:
        print(f"Found {total} drivers to evaluate")
        print("="*80)
    
    for idx, driver in enumerate(drivers, 1):
        driver_name = driver['name']
        gt_file = driver['gt_file']
        pred_file = driver['pred_file']
        
        # 获取驱动特定的参数（覆盖全局参数）
        driver_max_length = driver.get('max_length', max_length)
        driver_use_sampling = driver.get('use_sampling', use_sampling)
        driver_sample_size = driver.get('sample_size', sample_size)
        driver_max_sequences = driver.get('max_sequences', max_sequences)
        
        if verbose:
            print(f"\n[{idx}/{total}] Evaluating driver: {driver_name}")
            print("-"*80)
        
        try:
            # 检查文件是否存在
            if not Path(gt_file).exists():
                print(f"Warning: GT file not found: {gt_file}", file=sys.stderr)
                continue
            
            if not Path(pred_file).exists():
                print(f"Warning: Prediction file not found: {pred_file}", file=sys.stderr)
                continue
            
            # 执行评估
            result = evaluate_graphs(
                gt_file=gt_file,
                pred_file=pred_file,
                max_length=driver_max_length,
                use_sampling=driver_use_sampling,
                sample_size=driver_sample_size,
                max_sequences=driver_max_sequences,
                verbose=verbose
            )
            
            # 添加驱动名称
            result['driver_name'] = driver_name
            result['gt_file'] = gt_file
            result['pred_file'] = pred_file
            
            # 保存单个驱动的结果
            driver_output_file = output_path / f"{driver_name}_result.json"
            save_results_json(result, str(driver_output_file))
            
            if verbose:
                print(f"\nResult saved to: {driver_output_file}")
                print(f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1_score']:.4f}")
            
            # 添加到结果列表
            all_results.append(result)
            
            # 添加到摘要列表（只包含主要指标）
            summary_results.append({
                'driver_name': driver_name,
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'gt_sequences': result['gt_sequences'],
                'pred_sequences': result['pred_sequences'],
                'common_sequences': result['common_sequences'],
            })
            
        except Exception as e:
            print(f"Error evaluating driver {driver_name}: {e}", file=sys.stderr)
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # 保存汇总结果
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format in ['json', 'both']:
            # 保存完整结果（JSON）
            full_output_file = output_path / f"all_results_{timestamp}.json"
            save_results_json(all_results, str(full_output_file))
            if verbose:
                print(f"\nAll results saved to: {full_output_file}")
        
        if output_format in ['csv', 'both']:
            # 保存摘要结果（CSV）
            csv_output_file = output_path / f"summary_{timestamp}.csv"
            save_results_csv(summary_results, str(csv_output_file))
            if verbose:
                print(f"Summary saved to: {csv_output_file}")
        
        # 计算平均指标
        if len(summary_results) > 0:
            avg_precision = sum(r['precision'] for r in summary_results) / len(summary_results)
            avg_recall = sum(r['recall'] for r in summary_results) / len(summary_results)
            avg_f1 = sum(r['f1_score'] for r in summary_results) / len(summary_results)
            
            if verbose:
                print("\n" + "="*80)
                print("Overall Summary")
                print("="*80)
                print(f"Total drivers evaluated: {len(summary_results)}")
                print(f"Average Precision: {avg_precision:.4f}")
                print(f"Average Recall:    {avg_recall:.4f}")
                print(f"Average F1 Score:   {avg_f1:.4f}")
                print("="*80)
    else:
        print("No results to save", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='批量评估多个驱动的ioctl状态转换图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_evaluator.py config.json
  python batch_evaluator.py config.json --output-dir results --format csv
  python batch_evaluator.py config.json --output-dir results --format both --quiet
        """
    )
    
    parser.add_argument('config_file', help='配置文件路径（JSON格式）')
    parser.add_argument('--output-dir', '-o', default='results',
                       help='输出目录 (默认: results)')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'both'],
                       default='both', help='输出格式 (默认: both)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式，不输出详细信息')
    
    args = parser.parse_args()
    
    batch_evaluate(
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_format=args.format,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
