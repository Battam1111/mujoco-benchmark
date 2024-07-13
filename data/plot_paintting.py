import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def read_csv_files(file_paths: List[str]) -> List[pd.DataFrame]:
    """读取CSV文件列表并返回DataFrame列表。"""
    return [pd.read_csv(file) for file in file_paths]

def compute_mean_and_std(dfs: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算DataFrame列表的均值和标准差。"""
    combined_df = pd.concat(dfs)
    grouped = combined_df.groupby('epoch')
    mean_df = grouped.mean()
    std_df = grouped.std()
    return mean_df, std_df

def plot_data(df: pd.DataFrame, std_df: pd.DataFrame, columns: List[str], xlabel: str, ylabel: str, title: str, output_file: str, window_size: int = 100, plot_std: bool = True):
    """绘制数据图表并保存到文件。"""
    plt.figure()
    for column in columns:
        plt.plot(df.index, df[column], label=f"{column} (Mean)")
        if plot_std:
            plt.fill_between(df.index, df[column] - std_df[column], df[column] + std_df[column], alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    # 绘制平滑数据图
    plt.figure()
    for column in columns:
        smooth_mean = df[column].rolling(window=window_size).mean()
        if plot_std:
            smooth_std = std_df[column].rolling(window=window_size).mean()
            plt.fill_between(df.index, smooth_mean - smooth_std, smooth_mean + smooth_std, alpha=0.2)
        plt.plot(df.index, smooth_mean, label=f"{column} (Smoothed Mean)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Smoothed {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file.replace('.png', '_smoothed.png'))
    plt.close()

def extract_info_from_path(file_path: str) -> Tuple[str, str, str]:
    """从文件路径中提取任务名称、数据类型和种子。"""
    parts = file_path.split('/')
    task_name = parts[-3]
    data_type = parts[-2]
    seed = parts[-1].split('_')[0]
    return task_name, data_type, seed

def plot_single_experiment(csv_file_TR: str, csv_file_TL: str, output_dir: str, window_size: int = 100):
    """读取单个实验的CSV文件并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)
    df_rewards = pd.read_csv(csv_file_TR)
    df_lengths = pd.read_csv(csv_file_TL)

    task_name, data_type_TR, seed_TR = extract_info_from_path(csv_file_TR)
    _, data_type_TL, seed_TL = extract_info_from_path(csv_file_TL)

    reward_columns = df_rewards.columns[1:3]
    length_columns = df_lengths.columns[1:3]

    plot_data(df_rewards, pd.DataFrame(), reward_columns, 'Epoch', 'Average Reward', 'Average Reward vs Epoch', os.path.join(output_dir, f'{task_name}_{data_type_TR}_{seed_TR}_average_reward.png'), window_size, plot_std=False)
    plot_data(df_lengths, pd.DataFrame(), length_columns, 'Epoch', 'Episode Length', 'Episode Length vs Epoch', os.path.join(output_dir, f'{task_name}_{data_type_TL}_{seed_TL}_episode_length.png'), window_size, plot_std=False)

def plot_multiple_experiments(csv_files_TR: List[str], csv_files_TL: List[str], output_dir: str, window_size: int = 100):
    """读取多个实验的CSV文件，计算均值和标准差，并绘制图表。"""
    os.makedirs(output_dir, exist_ok=True)

    dfs_rewards = read_csv_files(csv_files_TR)
    mean_rewards_df, std_rewards_df = compute_mean_and_std(dfs_rewards)

    dfs_lengths = read_csv_files(csv_files_TL)
    mean_lengths_df, std_lengths_df = compute_mean_and_std(dfs_lengths)

    task_name, data_type_TR, _ = extract_info_from_path(csv_files_TR[0])
    _, data_type_TL, _ = extract_info_from_path(csv_files_TL[0])
    seeds_TR = '-'.join([extract_info_from_path(f)[2] for f in csv_files_TR])
    seeds_TL = '-'.join([extract_info_from_path(f)[2] for f in csv_files_TL])

    reward_columns = mean_rewards_df.columns[:2]
    length_columns = mean_lengths_df.columns[:2]

    plot_data(mean_rewards_df, std_rewards_df, reward_columns, 'Epoch', 'Average Reward', 'Average Reward vs Epoch', os.path.join(output_dir, f'{task_name}_{data_type_TR}_Seeds({seeds_TR})_average_reward.png'), window_size)
    plot_data(mean_lengths_df, std_lengths_df, length_columns, 'Epoch', 'Episode Length', 'Episode Length vs Epoch', os.path.join(output_dir, f'{task_name}_{data_type_TL}_Seeds({seeds_TL})_episode_length.png'), window_size)

# 使用示例
single_csv_file_TR = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed4_20240713-131533.csv'
single_csv_file_TL = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed4_20240713-131533.csv'
output_dir_single = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/plot_set/single'

multiple_csv_files_TR = [
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed0_20240712-201616.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed1_20240713-004009.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed2_20240713-045031.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed3_20240713-090818.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestRew/Seed4_20240713-131533.csv',
]
multiple_csv_files_TL = [
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed0_20240712-201616.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed1_20240713-004009.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed2_20240713-045031.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed3_20240713-090818.csv',
    '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/Humanoid-v4/TestLen/Seed4_20240713-131533.csv',
]
output_dir_multiple = '/home/eias/Desktop/YJ-SACRefine/mujoco-benchmark/data/plot_set/multiple'

# 绘制单个实验数据的图表
plot_single_experiment(single_csv_file_TR, single_csv_file_TL, output_dir_single)

# 绘制多个实验数据的图表
plot_multiple_experiments(multiple_csv_files_TR, multiple_csv_files_TL, output_dir_multiple)
