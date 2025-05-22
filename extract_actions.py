import h5py
import os
import argparse

def extract_actions(input_file, output_file_left, output_file_right):
    """
    从原始hdf5文件中提取动作数据并保存到新的hdf5文件中
    
    Args:
        input_file (str): 输入hdf5文件的路径
        output_file (str): 输出hdf5文件的路径
    """
    print(f"正在从 {input_file} 提取动作数据...")
    
    with h5py.File(input_file, 'r') as f:
        actions = f['action'][()][:,:]
    
    with h5py.File(output_file_left, 'w') as f:
        f.create_dataset('action', data=actions[:, 0:7])
    with h5py.File(output_file_right, 'w') as f:
        f.create_dataset('action', data=actions[:, 7:14])
    
    print(f"动作数据已保存到 {output_file_left} 和 {output_file_right}")
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
        output_filename_left = f"actions_left_{input_filename}"
        output_filename_right = f"actions_right_{input_filename}"
        args.output_file_left = os.path.join(input_dir, output_filename_left)
        args.output_file_right = os.path.join(input_dir, output_filename_right)
    
    extract_actions(args.input_file, args.output_file_left, args.output_file_right)

if __name__ == "__main__":
    main() 