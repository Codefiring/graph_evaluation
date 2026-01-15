#!/usr/bin/env python3
"""
生成测试数据：用于测试基于拓扑结构的图编辑评估算法
- 第二列（ioctl）在gt和pred中相同
- 第一、三列（状态）在gt和pred中表达不同，但拓扑结构相同
"""

import csv
from pathlib import Path

def generate_test_data_small():
    """生成小规模测试数据（3-5条）"""
    # GT数据：简单的线性状态转换
    gt_data = [
        ("STATE_A", "open", "STATE_B"),
        ("STATE_B", "read", "STATE_C"),
        ("STATE_C", "close", "STATE_D"),
    ]
    
    # Pred数据：相同的拓扑结构，但状态名称不同
    pred_data = [
        ("S1", "open", "S2"),
        ("S2", "read", "S3"),
        ("S3", "close", "S4"),
    ]
    
    return gt_data, pred_data

def generate_test_data_medium():
    """生成中等规模测试数据（10-15条）"""
    # GT数据：有分支的状态转换图
    gt_data = [
        ("INIT_STATE", "init", "READY"),
        ("READY", "open", "OPENED"),
        ("OPENED", "read", "READING"),
        ("OPENED", "write", "WRITING"),
        ("READING", "seek", "SEEKING"),
        ("READING", "close", "CLOSED"),
        ("WRITING", "flush", "FLUSHED"),
        ("WRITING", "close", "CLOSED"),
        ("SEEKING", "read", "READING"),
        ("SEEKING", "close", "CLOSED"),
        ("FLUSHED", "close", "CLOSED"),
    ]
    
    # Pred数据：相同的拓扑结构，但状态名称不同
    pred_data = [
        ("START", "init", "READY_STATE"),
        ("READY_STATE", "open", "OPEN_STATE"),
        ("OPEN_STATE", "read", "READ_STATE"),
        ("OPEN_STATE", "write", "WRITE_STATE"),
        ("READ_STATE", "seek", "SEEK_STATE"),
        ("READ_STATE", "close", "END_STATE"),
        ("WRITE_STATE", "flush", "FLUSH_STATE"),
        ("WRITE_STATE", "close", "END_STATE"),
        ("SEEK_STATE", "read", "READ_STATE"),
        ("SEEK_STATE", "close", "END_STATE"),
        ("FLUSH_STATE", "close", "END_STATE"),
    ]
    
    return gt_data, pred_data

def generate_test_data_large():
    """生成大规模测试数据（20多条）"""
    # GT数据：复杂的状态转换图，包含多个分支和循环
    gt_data = [
        ("INIT", "start", "IDLE"),
        ("IDLE", "connect", "CONNECTING"),
        ("IDLE", "listen", "LISTENING"),
        ("CONNECTING", "success", "CONNECTED"),
        ("CONNECTING", "fail", "IDLE"),
        ("LISTENING", "accept", "ACCEPTED"),
        ("LISTENING", "cancel", "IDLE"),
        ("CONNECTED", "send", "SENDING"),
        ("CONNECTED", "recv", "RECEIVING"),
        ("CONNECTED", "close", "CLOSING"),
        ("ACCEPTED", "send", "SENDING"),
        ("ACCEPTED", "recv", "RECEIVING"),
        ("ACCEPTED", "close", "CLOSING"),
        ("SENDING", "done", "CONNECTED"),
        ("SENDING", "done", "ACCEPTED"),
        ("SENDING", "error", "ERROR"),
        ("RECEIVING", "done", "CONNECTED"),
        ("RECEIVING", "done", "ACCEPTED"),
        ("RECEIVING", "error", "ERROR"),
        ("ERROR", "reset", "IDLE"),
        ("ERROR", "close", "CLOSING"),
        ("CLOSING", "finish", "CLOSED"),
        ("CLOSED", "restart", "IDLE"),
    ]
    
    # Pred数据：相同的拓扑结构，但状态名称不同
    pred_data = [
        ("BEGIN", "start", "WAIT"),
        ("WAIT", "connect", "CONN"),
        ("WAIT", "listen", "LISTEN"),
        ("CONN", "success", "LINKED"),
        ("CONN", "fail", "WAIT"),
        ("LISTEN", "accept", "ACCEPT"),
        ("LISTEN", "cancel", "WAIT"),
        ("LINKED", "send", "TX"),
        ("LINKED", "recv", "RX"),
        ("LINKED", "close", "SHUTDOWN"),
        ("ACCEPT", "send", "TX"),
        ("ACCEPT", "recv", "RX"),
        ("ACCEPT", "close", "SHUTDOWN"),
        ("TX", "done", "LINKED"),
        ("TX", "done", "ACCEPT"),
        ("TX", "error", "FAIL"),
        ("RX", "done", "LINKED"),
        ("RX", "done", "ACCEPT"),
        ("RX", "error", "FAIL"),
        ("FAIL", "reset", "WAIT"),
        ("FAIL", "close", "SHUTDOWN"),
        ("SHUTDOWN", "finish", "END"),
        ("END", "restart", "WAIT"),
    ]
    
    return gt_data, pred_data

def save_data_to_file(data, filepath):
    """将数据保存为CSV格式文件"""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for row in data:
            writer.writerow(row)

def generate_variants_with_differences():
    """生成带有拓扑差异的变体数据"""
    variants = {}
    
    # 1. 小规模：缺少边的情况
    gt_small = [
        ("STATE_A", "open", "STATE_B"),
        ("STATE_B", "read", "STATE_C"),
        ("STATE_C", "close", "STATE_D"),
    ]
    # Pred缺少一条边
    pred_small_missing = [
        ("S1", "open", "S2"),
        ("S2", "read", "S3"),
        # 缺少 ("S3", "close", "S4")
    ]
    variants['small_missing'] = (gt_small, pred_small_missing, "小规模-缺少边")
    
    # 2. 小规模：多余边的情况
    pred_small_extra = [
        ("S1", "open", "S2"),
        ("S2", "read", "S3"),
        ("S3", "close", "S4"),
        ("S2", "write", "S5"),  # 多余的边
    ]
    variants['small_extra'] = (gt_small, pred_small_extra, "小规模-多余边")
    
    # 3. 中等规模：缺少边的情况
    gt_medium = [
        ("INIT_STATE", "init", "READY"),
        ("READY", "open", "OPENED"),
        ("OPENED", "read", "READING"),
        ("OPENED", "write", "WRITING"),
        ("READING", "seek", "SEEKING"),
        ("READING", "close", "CLOSED"),
        ("WRITING", "flush", "FLUSHED"),
        ("WRITING", "close", "CLOSED"),
        ("SEEKING", "read", "READING"),
        ("SEEKING", "close", "CLOSED"),
        ("FLUSHED", "close", "CLOSED"),
    ]
    # Pred缺少一些边
    pred_medium_missing = [
        ("START", "init", "READY_STATE"),
        ("READY_STATE", "open", "OPEN_STATE"),
        ("OPEN_STATE", "read", "READ_STATE"),
        ("OPEN_STATE", "write", "WRITE_STATE"),
        # 缺少 ("READ_STATE", "seek", "SEEK_STATE")
        ("READ_STATE", "close", "END_STATE"),
        ("WRITE_STATE", "flush", "FLUSH_STATE"),
        ("WRITE_STATE", "close", "END_STATE"),
        # 缺少 ("SEEK_STATE", "read", "READ_STATE")
        # 缺少 ("SEEK_STATE", "close", "END_STATE")
        ("FLUSH_STATE", "close", "END_STATE"),
    ]
    variants['medium_missing'] = (gt_medium, pred_medium_missing, "中等规模-缺少边")
    
    # 4. 中等规模：多余边的情况
    pred_medium_extra = [
        ("START", "init", "READY_STATE"),
        ("READY_STATE", "open", "OPEN_STATE"),
        ("OPEN_STATE", "read", "READ_STATE"),
        ("OPEN_STATE", "write", "WRITE_STATE"),
        ("READ_STATE", "seek", "SEEK_STATE"),
        ("READ_STATE", "close", "END_STATE"),
        ("WRITE_STATE", "flush", "FLUSH_STATE"),
        ("WRITE_STATE", "close", "END_STATE"),
        ("SEEK_STATE", "read", "READ_STATE"),
        ("SEEK_STATE", "close", "END_STATE"),
        ("FLUSH_STATE", "close", "END_STATE"),
        ("READ_STATE", "reset", "READY_STATE"),  # 多余的边
        ("WRITE_STATE", "reset", "READY_STATE"),  # 多余的边
    ]
    variants['medium_extra'] = (gt_medium, pred_medium_extra, "中等规模-多余边")
    
    # 5. 中等规模：混合情况（既有缺少的边，也有多余的边）
    pred_medium_mixed = [
        ("START", "init", "READY_STATE"),
        ("READY_STATE", "open", "OPEN_STATE"),
        ("OPEN_STATE", "read", "READ_STATE"),
        # 缺少 ("OPEN_STATE", "write", "WRITE_STATE")
        ("READ_STATE", "seek", "SEEK_STATE"),
        ("READ_STATE", "close", "END_STATE"),
        # 缺少 ("WRITE_STATE", "flush", "FLUSH_STATE")
        # 缺少 ("WRITE_STATE", "close", "END_STATE")
        ("SEEK_STATE", "read", "READ_STATE"),
        ("SEEK_STATE", "close", "END_STATE"),
        # 缺少 ("FLUSH_STATE", "close", "END_STATE")
        ("READ_STATE", "back", "OPEN_STATE"),  # 多余的边
    ]
    variants['medium_mixed'] = (gt_medium, pred_medium_mixed, "中等规模-混合差异")
    
    # 6. 大规模：缺少边的情况
    gt_large = [
        ("INIT", "start", "IDLE"),
        ("IDLE", "connect", "CONNECTING"),
        ("IDLE", "listen", "LISTENING"),
        ("CONNECTING", "success", "CONNECTED"),
        ("CONNECTING", "fail", "IDLE"),
        ("LISTENING", "accept", "ACCEPTED"),
        ("LISTENING", "cancel", "IDLE"),
        ("CONNECTED", "send", "SENDING"),
        ("CONNECTED", "recv", "RECEIVING"),
        ("CONNECTED", "close", "CLOSING"),
        ("ACCEPTED", "send", "SENDING"),
        ("ACCEPTED", "recv", "RECEIVING"),
        ("ACCEPTED", "close", "CLOSING"),
        ("SENDING", "done", "CONNECTED"),
        ("SENDING", "done", "ACCEPTED"),
        ("SENDING", "error", "ERROR"),
        ("RECEIVING", "done", "CONNECTED"),
        ("RECEIVING", "done", "ACCEPTED"),
        ("RECEIVING", "error", "ERROR"),
        ("ERROR", "reset", "IDLE"),
        ("ERROR", "close", "CLOSING"),
        ("CLOSING", "finish", "CLOSED"),
        ("CLOSED", "restart", "IDLE"),
    ]
    # Pred缺少一些边
    pred_large_missing = [
        ("BEGIN", "start", "WAIT"),
        ("WAIT", "connect", "CONN"),
        ("WAIT", "listen", "LISTEN"),
        ("CONN", "success", "LINKED"),
        # 缺少 ("CONN", "fail", "WAIT")
        ("LISTEN", "accept", "ACCEPT"),
        ("LISTEN", "cancel", "WAIT"),
        ("LINKED", "send", "TX"),
        ("LINKED", "recv", "RX"),
        ("LINKED", "close", "SHUTDOWN"),
        ("ACCEPT", "send", "TX"),
        ("ACCEPT", "recv", "RX"),
        ("ACCEPT", "close", "SHUTDOWN"),
        ("TX", "done", "LINKED"),
        # 缺少 ("TX", "done", "ACCEPT")
        ("TX", "error", "FAIL"),
        ("RX", "done", "LINKED"),
        ("RX", "done", "ACCEPT"),
        ("RX", "error", "FAIL"),
        ("FAIL", "reset", "WAIT"),
        ("FAIL", "close", "SHUTDOWN"),
        ("SHUTDOWN", "finish", "END"),
        ("END", "restart", "WAIT"),
    ]
    variants['large_missing'] = (gt_large, pred_large_missing, "大规模-缺少边")
    
    # 7. 大规模：多余边的情况
    pred_large_extra = [
        ("BEGIN", "start", "WAIT"),
        ("WAIT", "connect", "CONN"),
        ("WAIT", "listen", "LISTEN"),
        ("CONN", "success", "LINKED"),
        ("CONN", "fail", "WAIT"),
        ("LISTEN", "accept", "ACCEPT"),
        ("LISTEN", "cancel", "WAIT"),
        ("LINKED", "send", "TX"),
        ("LINKED", "recv", "RX"),
        ("LINKED", "close", "SHUTDOWN"),
        ("ACCEPT", "send", "TX"),
        ("ACCEPT", "recv", "RX"),
        ("ACCEPT", "close", "SHUTDOWN"),
        ("TX", "done", "LINKED"),
        ("TX", "done", "ACCEPT"),
        ("TX", "error", "FAIL"),
        ("RX", "done", "LINKED"),
        ("RX", "done", "ACCEPT"),
        ("RX", "error", "FAIL"),
        ("FAIL", "reset", "WAIT"),
        ("FAIL", "close", "SHUTDOWN"),
        ("SHUTDOWN", "finish", "END"),
        ("END", "restart", "WAIT"),
        ("WAIT", "timeout", "WAIT"),  # 多余的边（自循环）
        ("LINKED", "ping", "LINKED"),  # 多余的边（自循环）
    ]
    variants['large_extra'] = (gt_large, pred_large_extra, "大规模-多余边")
    
    return variants


def main():
    """生成所有测试数据"""
    data_dir = Path("data/topology_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成完美匹配的数据（作为基准）
    print("Generating perfect match test data...")
    gt_small, pred_small = generate_test_data_small()
    save_data_to_file(gt_small, data_dir / "small_gt.txt")
    save_data_to_file(pred_small, data_dir / "small_pred.txt")
    print(f"  Saved: {data_dir / 'small_gt.txt'} ({len(gt_small)} transitions)")
    print(f"  Saved: {data_dir / 'small_pred.txt'} ({len(pred_small)} transitions)")
    
    gt_medium, pred_medium = generate_test_data_medium()
    save_data_to_file(gt_medium, data_dir / "medium_gt.txt")
    save_data_to_file(pred_medium, data_dir / "medium_pred.txt")
    print(f"  Saved: {data_dir / 'medium_gt.txt'} ({len(gt_medium)} transitions)")
    print(f"  Saved: {data_dir / 'medium_pred.txt'} ({len(pred_medium)} transitions)")
    
    gt_large, pred_large = generate_test_data_large()
    save_data_to_file(gt_large, data_dir / "large_gt.txt")
    save_data_to_file(pred_large, data_dir / "large_pred.txt")
    print(f"  Saved: {data_dir / 'large_gt.txt'} ({len(gt_large)} transitions)")
    print(f"  Saved: {data_dir / 'large_pred.txt'} ({len(pred_large)} transitions)")
    
    # 生成带有拓扑差异的变体数据
    print("\nGenerating variants with topology differences...")
    variants = generate_variants_with_differences()
    
    for variant_name, (gt_data, pred_data, description) in variants.items():
        # 确定使用哪个GT文件
        if 'small' in variant_name:
            gt_file = data_dir / "small_gt.txt"
        elif 'medium' in variant_name:
            gt_file = data_dir / "medium_gt.txt"
        elif 'large' in variant_name:
            gt_file = data_dir / "large_gt.txt"
        else:
            continue
        
        # 保存pred文件
        pred_file = data_dir / f"{variant_name}_pred.txt"
        save_data_to_file(pred_data, pred_file)
        print(f"  Saved: {pred_file} ({len(pred_data)} transitions) - {description}")
    
    print("\nAll test data generated successfully!")
    print(f"Data directory: {data_dir.absolute()}")

if __name__ == "__main__":
    main()
