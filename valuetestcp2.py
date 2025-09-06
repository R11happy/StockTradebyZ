import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.ticker as mticker
import os
import platform

# Suppress FutureWarning for pandas downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

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

def get_roe_pb_data(ts_code):
    """
    获取公司近5年ROE和PB数据
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')

    # 1. 获取ROE数据 (季度)
    try:
        roe_df = pro.fina_indicator(ts_code=ts_code, start_date=start_date_str, end_date=end_date_str, fields='end_date,roe')
        if roe_df is None or roe_df.empty or 'roe' not in roe_df.columns or roe_df['roe'].isnull().all():
            print(f"未找到 {ts_code} 从 {start_date_str} 到 {end_date_str} 的ROE数据")
            return pd.DataFrame()
        roe_df['end_date'] = pd.to_datetime(roe_df['end_date'])
        roe_df = roe_df.set_index('end_date').sort_index()
        roe_df = roe_df.dropna(subset=['roe'])
    except Exception as e:
        print(f"获取 {ts_code} 的ROE数据时出错: {e}")
        return pd.DataFrame()

    # 2. 获取PB数据 (每日)
    try:
        pb_df = pro.daily_basic(ts_code=ts_code, start_date=start_date_str, end_date=end_date_str, fields='trade_date,pb')
        if pb_df is None or pb_df.empty or 'pb' not in pb_df.columns or pb_df['pb'].isnull().all():
            print(f"未找到 {ts_code} 从 {start_date_str} 到 {end_date_str} 的PB数据")
            return pd.DataFrame()
        pb_df['trade_date'] = pd.to_datetime(pb_df['trade_date'])
        pb_df = pb_df.set_index('trade_date').sort_index()
        pb_df = pb_df.dropna(subset=['pb'])
    except Exception as e:
        print(f"获取 {ts_code} 的PB数据时出错: {e}")
        return pd.DataFrame()

    # 3. 合并ROE和PB数据
    # 使用asof合并，为每个ROE报告期找到最近的PB值
    combined_df = pd.merge_asof(roe_df.sort_index(), pb_df.sort_index(), left_index=True, right_index=True, direction='backward')
    
    if combined_df.empty or 'pb' not in combined_df.columns or combined_df['pb'].isnull().all():
        print(f"无法为 {ts_code} 对齐PB数据。")
        return pd.DataFrame()

    # 计算ROE/PB
    combined_df['roe_div_pb'] = combined_df['roe'] / combined_df['pb']
    combined_df['roe_div_pb'] = combined_df['roe_div_pb'].replace([float('inf'), -float('inf')], pd.NA).fillna(pd.NA)
    
    return combined_df[['roe', 'pb', 'roe_div_pb']]

def plot_roe_pb_trends(stock_data_dict):
    """
    绘制多支股票的ROE/PB趋势图
    """
    if not stock_data_dict:
        print("没有数据可用于绘制图表。")
        return

    fig, ax = plt.subplots(figsize=(14, 7)) # 单个子图

    for ts_code, df in stock_data_dict.items():
        company_name = get_company_name(ts_code)
        if not df.empty:
            # 绘制ROE/PB趋势
            if 'roe_div_pb' in df.columns and not df['roe_div_pb'].isnull().all():
                ax.plot(df.index, df['roe_div_pb'], label=f'{company_name} ROE/PB', marker='o', markersize=3, linewidth=1)
            else:
                print(f"Warning: No valid ROE/PB data for {company_name} ({ts_code}).")

    ax.set_xlabel('日期')
    ax.set_ylabel('ROE/PB')
    ax.set_title('近五年ROE/PB趋势')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle='--', alpha=0.7)

    # 自动调整日期刻度
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局以适应图例

    # 保存图片
    output_dir = 'valuItem'
    os.makedirs(output_dir, exist_ok=True)

    latest_date = datetime.now().strftime('%Y%m%d')
    filename = f"roe_pb_trends_{latest_date}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"图表已保存至: {filepath}")
    plt.show()

if __name__ == '__main__':
    stock_inputs = input("请输入股票代码，多个代码请用逗号分隔 (例如: 000001,600519): ")
    if not stock_inputs:
        print("股票代码不能为空。")
    else:
        # 检查Tushare token是否已设置
        if PRO_API_TOKEN == 'YOUR_TUSHARE_TOKEN':
            print("请在脚本中设置您的Tushare PRO_API_TOKEN。")
        else:
            ts_codes = []
            for stock_input in stock_inputs.split(','):
                stock_input = stock_input.strip()
                if '.' not in stock_input:
                    if stock_input.startswith('6'):
                        ts_codes.append(f"{stock_input}.SH")
                    elif stock_input.startswith('0') or stock_input.startswith('3'):
                        ts_codes.append(f"{stock_input}.SZ")
                    else:
                        print(f"无法自动识别股票代码 {stock_input} 的交易所，请手动输入完整代码 (例如: 000001.SZ)。")
                        continue
                else:
                    ts_codes.append(stock_input)

            if not ts_codes:
                print("没有有效的股票代码可供查询。")
            else:
                all_stock_data = {}
                for code in ts_codes:
                    data = get_roe_pb_data(code)
                    if not data.empty:
                        all_stock_data[code] = data
                
                if all_stock_data:
                    print(f"准备绘制以下股票的数据: {', '.join([get_company_name(code) for code in all_stock_data.keys()])}")
                else:
                    print("没有有效的股票数据可用于绘制图表。")
                plot_roe_pb_trends(all_stock_data)
