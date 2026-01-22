import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json

def read_tensorboard_logs(log_dir):
    """
    读取指定目录下的所有 tensorboard 日志文件
    
    Args:
        log_dir: tensorboard 日志文件夹路径
        
    Returns:
        dict: 包含所有实验数据的字典
    """
    log_path = Path(log_dir)
    results = {}
    
    # 遍历所有子目录
    for experiment_dir in log_path.rglob('*'):
        if experiment_dir.is_dir():
            # 查找该目录下的所有 tfevents 文件
            tfevents_files = list(experiment_dir.glob('events.out.tfevents.*'))
            
            if tfevents_files:
                experiment_name = str(experiment_dir.relative_to(log_path))
                print(f"处理实验: {experiment_name}")
                
                experiment_data = {
                    'scalars': {},
                    'images': {},
                    'histograms': {},
                    'metadata': {
                        'path': str(experiment_dir),
                        'files': [str(f) for f in tfevents_files]
                    }
                }
                
                # 加载每个 tfevents 文件
                for tfevents_file in tfevents_files:
                    try:
                        ea = event_accumulator.EventAccumulator(str(tfevents_file))
                        ea.Reload()
                        
                        # 读取标量数据
                        for tag in ea.Tags()['scalars']:
                            events = ea.Scalars(tag)
                            if tag not in experiment_data['scalars']:
                                experiment_data['scalars'][tag] = []
                            
                            for event in events:
                                experiment_data['scalars'][tag].append({
                                    'step': event.step,
                                    'value': event.value,
                                    'wall_time': event.wall_time
                                })
                        
                        # 读取图像标签（如果需要）
                        for tag in ea.Tags()['images']:
                            if tag not in experiment_data['images']:
                                experiment_data['images'][tag] = len(ea.Images(tag))
                        
                        # 读取直方图标签（如果需要）
                        for tag in ea.Tags()['histograms']:
                            if tag not in experiment_data['histograms']:
                                experiment_data['histograms'][tag] = len(ea.Histograms(tag))
                        
                        print(f"  - 成功读取: {tfevents_file.name}")
                        print(f"    标量指标: {list(experiment_data['scalars'].keys())}")
                        
                    except Exception as e:
                        print(f"  - 读取失败 {tfevents_file.name}: {e}")
                
                results[experiment_name] = experiment_data
    
    return results


def save_to_csv(data, output_dir):
    """
    将 tensorboard 数据保存为 CSV 文件
    
    Args:
        data: read_tensorboard_logs 返回的数据
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for experiment_name, experiment_data in data.items():
        # 为每个实验创建一个子目录
        exp_dir = output_path / experiment_name.replace('/', '_')
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存每个标量指标
        for tag, values in experiment_data['scalars'].items():
            if values:
                df = pd.DataFrame(values)
                csv_filename = f"{tag.replace('/', '_')}.csv"
                csv_path = exp_dir / csv_filename
                df.to_csv(csv_path, index=False)
                print(f"保存: {csv_path}")


def save_to_json(data, output_file):
    """
    将 tensorboard 数据保存为 JSON 文件
    
    Args:
        data: read_tensorboard_logs 返回的数据
        output_file: 输出 JSON 文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"保存 JSON: {output_file}")


def print_summary(data):
    """
    打印数据摘要
    
    Args:
        data: read_tensorboard_logs 返回的数据
    """
    print("\n" + "="*80)
    print("实验数据摘要")
    print("="*80)
    
    for experiment_name, experiment_data in data.items():
        print(f"\n实验: {experiment_name}")
        print(f"  文件数量: {len(experiment_data['metadata']['files'])}")
        print(f"  标量指标数量: {len(experiment_data['scalars'])}")
        
        for tag, values in experiment_data['scalars'].items():
            if values:
                print(f"    - {tag}: {len(values)} 个数据点")
                if len(values) > 0:
                    print(f"      步数范围: {values[0]['step']} - {values[-1]['step']}")
                    print(f"      值范围: {min(v['value'] for v in values):.4f} - {max(v['value'] for v in values):.4f}")


if __name__ == "__main__":
    # 配置路径
    tensorboard_log_dir = "/home/ranhengwang/ndsl-project/RL-Factory/tensorboard_log"
    output_csv_dir = "/home/ranhengwang/ndsl-project/RL-Factory/tensorboard_data_csv"
    output_json_file = "/home/ranhengwang/ndsl-project/RL-Factory/tensorboard_data.json"
    
    print("开始读取 TensorBoard 日志...")
    print(f"日志目录: {tensorboard_log_dir}\n")
    
    # 读取数据
    data = read_tensorboard_logs(tensorboard_log_dir)
    
    # 打印摘要
    print_summary(data)
    
    # 保存为 CSV
    print("\n保存数据为 CSV 格式...")
    save_to_csv(data, output_csv_dir)
    
    # 保存为 JSON
    print("\n保存数据为 JSON 格式...")
    save_to_json(data, output_json_file)
    
    print("\n完成!")