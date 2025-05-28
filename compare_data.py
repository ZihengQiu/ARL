import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse

def load_data(data_path):
    """加载数据，支持多种格式: pkl, npz, h5"""
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.pkl':
        # 原始pkl格式
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    elif file_ext == '.npz':
        # NPZ格式
        data = np.load(data_path)
        return data
    
    elif file_ext == '.h5':
        # HDF5格式
        data_dict = {}
        with h5py.File(data_path, 'r') as f:
            # 遍历所有键
            for key in f.keys():
                if key == 'metadata':
                    continue
                    
                if key.endswith('_pickled'):
                    # 处理pickle化的数据
                    orig_key = key.replace('_pickled', '')
                    pickled_data = f[key][()]
                    data_dict[orig_key] = pickle.loads(pickled_data.tobytes())
                else:
                    # 处理普通数据
                    data_dict[key] = f[key][:]
        return data_dict
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")

def extract_joint_positions(data, high_freq=False):
    """从不同格式的数据中提取关节位置"""
    if high_freq:
        # 高频数据直接是一个关节位置数组
        if isinstance(data, np.ndarray) or isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict) and 'high_freq_joint_positions' in data:
            return np.array(data['high_freq_joint_positions'])
        else:
            raise ValueError("无法从高频数据中提取关节位置")
    else:
        # 低频数据可能有不同的格式
        if isinstance(data, list) and hasattr(data[0], 'joint_positions'):
            # 原始pickle格式
            return np.array([obs.joint_positions for obs in data])
        elif isinstance(data, dict) and 'joint_positions' in data:
            # 新的统一数据集格式
            return data['joint_positions']
        else:
            raise ValueError("无法从低频数据中提取关节位置")

def get_timestamps(data, high_freq=False):
    """获取时间戳"""
    if high_freq:
        # 高频数据的时间戳通常是从0开始的均匀时间步长
        if isinstance(data, np.ndarray) or isinstance(data, list):
            return np.arange(len(data)) / 1000.0  # 假设1000Hz
        elif isinstance(data, dict) and 'timestamp' in data:
            return data['timestamp']
        else:
            count = len(extract_joint_positions(data, high_freq=True))
            return np.arange(count) / 1000.0
    else:
        # 低频数据的时间戳
        if isinstance(data, dict) and 'timestamp' in data:
            return data['timestamp']
        else:
            count = len(extract_joint_positions(data, high_freq=False))
            return np.arange(count) * 0.05  # 假设20Hz

def compare_joint_data(high_freq_path, low_freq_path, output_dir='.'):
    """比较高频和低频关节数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载高频数据: {high_freq_path}")
    high_freq_data = load_data(high_freq_path)
    
    print(f"加载低频数据: {low_freq_path}")
    low_freq_data = load_data(low_freq_path)
    
    # 提取关节位置
    high_freq_positions = extract_joint_positions(high_freq_data, high_freq=True)
    low_freq_positions = extract_joint_positions(low_freq_data, high_freq=False)
    
    # 获取时间戳
    high_freq_time = get_timestamps(high_freq_data, high_freq=True)
    low_freq_time = get_timestamps(low_freq_data, high_freq=False)
    
    # 打印基本信息
    print(f"\n高频数据点总数: {len(high_freq_positions)}")
    print(f"低频数据点总数: {len(low_freq_positions)}")
    print(f"关节数量: {high_freq_positions.shape[1]}")
    
    # 计算统计信息
    print(f"\n高频数据 - 每个关节的范围:")
    for i in range(high_freq_positions.shape[1]):
        min_val = np.min(high_freq_positions[:, i])
        max_val = np.max(high_freq_positions[:, i])
        print(f"  关节 {i+1}: {min_val:.6f} 到 {max_val:.6f} rad")
    
    # 1. 绘制高频关节位置
    plt.figure(figsize=(12, 8))
    for joint_idx in range(high_freq_positions.shape[1]):
        plt.plot(high_freq_time, high_freq_positions[:, joint_idx], 
                label=f'Joint {joint_idx+1}')

    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (rad)')
    plt.title('Joint Positions (High Frequency)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'high_freq_joint_positions.png'))
    plt.close()
    
    # 2. 计算并绘制关节速度
    joint_velocities = np.diff(high_freq_positions, axis=0) / np.diff(high_freq_time)[:, None]
    
    plt.figure(figsize=(12, 8))
    for joint_idx in range(joint_velocities.shape[1]):
        plt.plot(high_freq_time[1:], joint_velocities[:, joint_idx], 
                label=f'Joint {joint_idx+1}')

    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.title('Joint Velocities (Derived from High Frequency Positions)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'high_freq_joint_velocities.png'))
    plt.close()
    
    # 3. 绘制对比图（每个关节一张图）
    for joint_idx in range(high_freq_positions.shape[1]):
        plt.figure(figsize=(12, 6))
        
        # 高频数据
        plt.plot(high_freq_time, high_freq_positions[:, joint_idx], 
                'b-', label='High Frequency Data', alpha=0.7)

        # 低频数据
        plt.plot(low_freq_time, low_freq_positions[:, joint_idx], 
                'ro-', label='Low Frequency Data', markersize=6)

        plt.xlabel('Time (s)')
        plt.ylabel(f'Joint {joint_idx+1} Position (rad)')
        plt.title(f'Joint {joint_idx+1}: High vs Low Frequency Data Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'joint{joint_idx+1}_comparison.png'))
        plt.close()
    
    # 4. 综合对比图 (只显示前3个关节，避免过于拥挤)
    plt.figure(figsize=(15, 10))
    
    for joint_idx in range(min(3, high_freq_positions.shape[1])):
        # 高频数据
        plt.plot(high_freq_time, high_freq_positions[:, joint_idx], 
                '-', label=f'Joint {joint_idx+1} (High Freq)', alpha=0.7)

        # 低频数据
        plt.plot(low_freq_time, low_freq_positions[:, joint_idx], 
                'o', label=f'Joint {joint_idx+1} (Low Freq)', markersize=6)

    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (rad)')
    plt.title('Comparison of High and Low Frequency Joint Position Data')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'multi_joint_comparison.png'))
    plt.close()
    
    print(f"\n图表已保存到: {output_dir}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较高频和低频关节数据")
    parser.add_argument('--high', type=str, required=True, 
                        help='高频数据文件路径 (.pkl, .npz, .h5)')
    parser.add_argument('--low', type=str, required=True,
                        help='低频数据文件路径 (.pkl, .npz, .h5)')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='输出目录')
    
    args = parser.parse_args()
    compare_joint_data(args.high, args.low, args.output)

#python compare_data.py --high /Users/zeen/study/arl/data/rlbench/high_freq_joint_positions.pkl --low/Users/zeen/study/arl/data/rlbench/low_dim_obs.h5 --output ./comparison_results