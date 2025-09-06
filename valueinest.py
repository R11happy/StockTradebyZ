import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import os
import platform

# Tushare token - PLEASE REPLACE WITH YOUR ACTUAL TOKEN
PRO_API_TOKEN = '60d29499510471150805842b1c7fc97e3a7ece2676b4ead1707f94d0'
ts.set_token(PRO_API_TOKEN)
pro = ts.pro_api()

# 设置matplotlib支持中文
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用SimHei
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统使用Arial Unicode MS
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Linux系统或其他，可能需要安装中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def get_company_name(ts_code):
    """
    根据股票代码获取公司名称
    """
    try:
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        company_name = df[df['ts_code'] == ts_code]['name'].iloc[0]
        return company_name
    except Exception as e:
        print(f"Error getting company name for {ts_code}: {e}")
        return ts_code

def get_market_cap_and_revenue_data(ts_code):
    """
    获取公司近10年总市值和总营收数据
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10 * 365)

    # 获取市值数据
    # pro.daily_basic 接口提供每日市值数据
    # 注意：tushare的daily_basic接口免费用户只能获取最近一年的数据，
    # 如果需要10年数据，需要升级tushare权限或使用其他数据源。
    # 这里为了演示，先获取尽可能多的数据。
    market_cap_df = pro.daily_basic(ts_code=ts_code, start_date=start_date.strftime('%Y%m%d'),
                                    end_date=end_date.strftime('%Y%m%d'),
                                    fields='trade_date,total_mv')
    if market_cap_df is None or market_cap_df.empty:
        print(f"No market cap data found for {ts_code} from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
        market_cap_df = pd.DataFrame(columns=['trade_date', 'total_mv'])
    else:
        market_cap_df['trade_date'] = pd.to_datetime(market_cap_df['trade_date'])
        market_cap_df = market_cap_df.set_index('trade_date').sort_index()
        market_cap_df['total_mv'] = market_cap_df['total_mv'] / 10000.0 # 转换为亿元

    # 获取营收数据
    # pro.income 接口提供季度营收数据
    revenue_df = pro.income(ts_code=ts_code, start_date=start_date.strftime('%Y%m%d'),
                            end_date=end_date.strftime('%Y%m%d'),
                            fields='end_date,total_revenue')
    if revenue_df is None or revenue_df.empty:
        print(f"No revenue data found for {ts_code} from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}")
        revenue_df = pd.DataFrame(columns=['end_date', 'total_revenue'])
    else:
        revenue_df['end_date'] = pd.to_datetime(revenue_df['end_date'])
        revenue_df = revenue_df.set_index('end_date').sort_index()
        revenue_df['total_revenue'] = revenue_df['total_revenue'] / 100000000.0 # 转换为亿元

    return market_cap_df, revenue_df

def plot_trends(ts_code, market_cap_df, revenue_df):
    """
    绘制总市值和总营收趋势图
    """
    company_name = get_company_name(ts_code)

    if market_cap_df.empty and revenue_df.empty:
        print(f"No data to plot for {company_name} ({ts_code}).")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制总营收柱状图
    if not revenue_df.empty:
        ax1.bar(revenue_df.index, revenue_df['total_revenue'], width=90, label='总营收', color='skyblue')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('总营收 (亿元)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f')) # 格式化Y轴标签为整数

    # 创建第二个Y轴，用于总市值
    ax2 = ax1.twinx()
    if not market_cap_df.empty:
        # 对总市值数据进行平滑处理
        market_cap_df['total_mv_smoothed'] = market_cap_df['total_mv'].rolling(window=30, min_periods=1).mean()
        ax2.plot(market_cap_df.index, market_cap_df['total_mv_smoothed'], label='总市值 (平滑)', color='red', linewidth=2)
        ax2.set_ylabel('总市值 (亿元)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f')) # 格式化Y轴标签为整数

    # 自适应Y轴范围
    if not revenue_df.empty:
        min_rev, max_rev = revenue_df['total_revenue'].min(), revenue_df['total_revenue'].max()
        ax1.set_ylim(bottom=0, top=max_rev * 1.2 if max_rev > 0 else 100) # 营收Y轴从0开始，顶部留20%空间
    if not market_cap_df.empty:
        min_mv, max_mv = market_cap_df['total_mv'].min(), market_cap_df['total_mv'].max()
        ax2.set_ylim(bottom=0, top=max_mv * 1.2 if max_mv > 0 else 100) # 市值Y轴从0开始，顶部留20%空间

    # 设置标题
    plt.title(f'{company_name}市值与业绩增长趋势')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # 自动调整日期刻度
    fig.autofmt_xdate()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图片
    output_dir = os.path.join('valuItem', ts_code.split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    latest_date = datetime.now().strftime('%Y%m%d')
    if not market_cap_df.empty:
        latest_date = market_cap_df.index.max().strftime('%Y%m%d')
    elif not revenue_df.empty:
        latest_date = revenue_df.index.max().strftime('%Y%m%d')

    filename = f"{latest_date}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"图表已保存至: {filepath}")
    plt.show()

if __name__ == '__main__':
    stock_input = input("请输入股票代码 (例如: 000001 或 600519): ")
    if not stock_input:
        print("股票代码不能为空。")
    else:
        # 检查Tushare token是否已设置
        if PRO_API_TOKEN == 'YOUR_TUSHARE_TOKEN':
            print("请在脚本中设置您的Tushare PRO_API_TOKEN。")
        else:
            # 自动补全股票代码
            if '.' not in stock_input:
                if stock_input.startswith('6'):
                    ts_code = f"{stock_input}.SH"
                elif stock_input.startswith('0') or stock_input.startswith('3'):
                    ts_code = f"{stock_input}.SZ"
                else:
                    print("无法自动识别股票代码的交易所，请手动输入完整代码 (例如: 000001.SZ)。")
                    exit()
            else:
                ts_code = stock_input

            market_cap_data, revenue_data = get_market_cap_and_revenue_data(ts_code)
            plot_trends(ts_code, market_cap_data, revenue_data)
