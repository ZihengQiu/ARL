import h5py
import os
import argparse

def extract_actions(input_file, output_file):
    """
    从原始hdf5文件中提取动作数据并保存到新的hdf5文件中
    
    Args:
        input_file (str): 输入hdf5文件的路径
        output_file (str): 输出hdf5文件的路径
    """
    print(f"正在从 {input_file} 提取动作数据...")
    
    # 读取原始文件中的动作数据
    with h5py.File(input_file, 'r') as f:
        actions = f['action'][()][:, 0:7]  # 只取前7维
    
    # 创建新的hdf5文件并保存动作数据
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('action', data=actions)
    
    print(f"动作数据已保存到 {output_file}")
    print(f"动作数据形状: {actions.shape}")

def main():
    parser = argparse.ArgumentParser(description='从原始hdf5文件中提取动作数据')
    parser.add_argument('input_file', help='输入hdf5文件的路径')
    parser.add_argument('--output_file', help='输出hdf5文件的路径（可选）')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，则在输入文件同目录下创建
    if args.output_file is None:
        input_dir = os.path.dirname(args.input_file)
        input_filename = os.path.basename(args.input_file)
        output_filename = f"actions_{input_filename}"
        args.output_file = os.path.join(input_dir, output_filename)
    
    extract_actions(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 