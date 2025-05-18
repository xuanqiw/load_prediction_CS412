import pandas as pd
import matplotlib.pyplot as plt
import warnings

# 1. 设置中文字体（确保系统已安装 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 只屏蔽缺少 glyph 的警告
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

# 然后运行你的绘图代码即可，比如：
# plt.plot(...)
# plt.title('第1周负荷时序图')
# plt.tight_layout()
# plt.show()
# 1. 读取数据并转换为长格式
df = pd.read_excel('train_data.xlsx', parse_dates=['Date'])
df_long = df.set_index('Date').stack().reset_index()
df_long.columns = ['Date', 'interval', 'load']

# 2. 计算每个 interval 对应的 timestamp
df_long['interval_num'] = df_long['interval'].str.replace('Power', '', regex=False).astype(int) - 1
df_long['timestamp'] = (
    df_long['Date'] +
    pd.to_timedelta(df_long['interval_num'] * 15, unit='m')
)
df_long = df_long.set_index('timestamp').sort_index()

# 3. 提取前三个完整周的数据
start = df_long.index.min().normalize()
week1 = df_long[start : start + pd.Timedelta(days=6)]
week2 = df_long[start + pd.Timedelta(days=7) : start + pd.Timedelta(days=13)]
week3 = df_long[start + pd.Timedelta(days=14) : start + pd.Timedelta(days=20)]

# 4. 绘制前三周负荷时序图
for i, week in enumerate((week1, week2, week3), start=1):
    plt.figure()
    plt.plot(week.index, week['load'])
    plt.title(f'第{i}周负荷时序图')
    plt.xlabel('时间')
    plt.ylabel('负荷')
    plt.tight_layout()

# 5. 按工作日/周末计算并绘制平均日负荷曲线
df_long['type_of_day'] = df_long.index.weekday.map(lambda x: '周末' if x >= 5 else '工作日')
df_long['hour_of_day'] = df_long.index.hour + df_long.index.minute / 60
daily_profile = df_long.groupby(['hour_of_day', 'type_of_day'])['load'].mean().unstack()

plt.figure()
for col in daily_profile.columns:
    plt.plot(daily_profile.index, daily_profile[col], label=col)
plt.title('工作日 vs 周末 平均日负荷曲线对比')
plt.xlabel('小时数')
plt.ylabel('平均负荷')
plt.legend(title='类型')
plt.tight_layout()

plt.show()