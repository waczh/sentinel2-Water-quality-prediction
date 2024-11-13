import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def plot_boxplots(data_3d):
    # 创建包含四个子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("box plot", fontsize=16)
    for station_id in range(4):
        ax = axes[station_id // 2, station_id % 2]
        # ax.set_title(f"station {station_id + 1}")
        # ax.set_xlabel("channel")
        # ax.set_ylabel("value")
        data_for_boxplot = [data_3d[station_id, channel_id, :] for channel_id in range(9)]
        ax.boxplot(data_for_boxplot, labels=[f"c {i + 1}" for i in range(9)])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_serial_data(array_3d):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("plot_serial_data", fontsize=16)
    for station_id in range(4):
        ax = axes[station_id // 2, station_id % 2]
        ax.set_xlabel("Time")
        ax.set_ylabel("value")
        for channel_id in range(9):
            ax.plot(array_3d[station_id, channel_id, :], label=f"channel {channel_id + 1}")
        # ax.legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def Encoding(filepath):
# 获取当前目录下的所有 .xlsx 文件，并按文件名中的日期排序
    files = [f for f in os.listdir(filepath) if f.endswith('.xlsx')]
    files.sort(key=lambda x: datetime.strptime(x.split('_')[-1][1:15], '%Y%m%dT%H%M%S'))
    data_list = []
    dates = []
    for file in files:
        # 提取日期信息
        date_str = file.split('_')[-1].split('T')[0]
        date = datetime.strptime(date_str, '%Y%m%d')
        df = pd.read_excel(file, usecols="A:I", nrows=4)
        data_array = df.to_numpy()
        if np.isnan(data_array).any(): continue
        dates.append(date)
        data_list.append(data_array)
    # 转换为 (4, 9, N) 的三维数组
    data_array_3d = np.stack(data_list, axis=2)
    plot_boxplots(data_array_3d)
    # 构建完整的日期范围
    date_range = pd.date_range(start=min(dates), end=max(dates), freq='D')
    # 对第三维度进行插值
    interpolated_data = np.empty((4, 9, len(date_range)))
    for i in range(4):  # 对每个监测站进行插值
        for j in range(9):  # 对每个通道进行插值
            original_values = data_array_3d[i, j, :]
            original_days = [(d - date_range[0]).days for d in dates]
            # 创建插值函数并进行插值
            interp_func = interp1d(original_days, original_values, kind='linear', fill_value="extrapolate")
            interpolated_values = interp_func([(d - date_range[0]).days for d in date_range])
            interpolated_data[i, j, :] = interpolated_values
    return interpolated_data

def gm11_predict(data):
    """
    GM(1,1) 灰色预测模型，用于预测序列的下一时刻值。
    """
    N = len(data)
    x1 = np.cumsum(data)  # 累加生成序列
    B = np.column_stack((-0.5 * (x1[:-1] + x1[1:]), np.ones(N - 1)))  # 构造数据矩阵
    Y = data[1:].reshape(-1, 1)
    [[a], [b]] = np.linalg.inv(B.T @ B) @ B.T @ Y
    next_value = (data[0] - b / a) * np.exp(-a * N) + b / a
    return next_value

def predict_next_values(data_3d):
    # 初始化输出数组 (4, 9, 1)
    output = np.empty((4, 9, 1))
    # 循环每个监测站
    for station_id in range(4):
        # 循环每个通道
        for channel_id in range(9):
            # 获取对应通道的 N 个值
            data_series = data_3d[station_id, channel_id, :]
            # 利用 GM(1,1) 进行预测
            next_value = gm11_predict(data_series)
            # 存储预测结果
            output[station_id, channel_id, 0] = next_value
    return output

def GM11(data ,timestep):

     for idx in range(timestep):
         output_current = predict_next_values(data)
         data = np.concatenate((data, output_current), axis=2)
     return data

if __name__=='__main__':
    data = Encoding('.')
    plot_serial_data(data)
    time_step = 1
    output = GM11(data[:,:,-5:],time_step)

