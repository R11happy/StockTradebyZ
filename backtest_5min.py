from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from select_stock import instantiate_selector, load_config
# from Selector import Selector # Assuming Selector.py contains the base Selector class
import Selector # Import the entire module to access functions if needed

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest_5min") # Changed logger name

def load_data_5min(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        # The CSV files have the datetime as the index.
        df = pd.read_csv(fp, index_col=0, parse_dates=True).sort_index()
        # Rename columns to match the expected 'date' and 'open', 'close', 'high', 'low', 'volume'
        df = df.rename(columns={
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "最新价": "close", # Use latest price as close if '收盘' is not available or for consistency
        })
        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        frames[code] = df
    return frames


def run_backtest(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    selector_alias: str,
    selector,
    all_data: dict[str, pd.DataFrame],
    hold_period: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    initial_capital: float,
    volume_ratio_threshold: float, # New parameter for volume ratio
):
    """执行回测"""
    trade_logs = []
    all_dates = sorted(list(set(d.date() for df in all_data.values() for d in df.index))) # Use index for dates
    trade_dates = [pd.Timestamp(d) for d in all_dates if start_date.date() <= d <= end_date.date()]
    
    # 用于跟踪持仓信息
    open_positions = {}
    
    # 用于跟踪每日的资金变化
    daily_capital_changes = []
    current_capital = initial_capital

    # Instantiate VolumeRatioSelector
    volume_selector = Selector.VolumeRatioSelector(ratio=volume_ratio_threshold)

    for i, current_date in enumerate(tqdm(trade_dates, desc=f"回测 {selector_alias}")):
        # 1. 选股 (first by the main selector, then by volume ratio)
        initial_picks = selector.select(current_date, all_data)
        
        # Filter picks based on volume ratio
        volume_ratio_picks = volume_selector.select(current_date, {code: all_data[code] for code in initial_picks})
        picks = [code for code in initial_picks if code in volume_ratio_picks]

        # 2. 模拟交易 - 买入
        if picks:
            # For 5min data, the "buy_date" should be the current_date, and the "buy_price"
            # should be the open price of the first 5min bar on that day.
            # We need to adjust the logic to use 5min data correctly.
            # For simplicity, let's assume "buy_date" is the current_date and "buy_price"
            # is the open of the first 5min bar of the day.
            
            # Calculate investment per stock based on available capital and number of picks
            num_picks_to_invest = len(picks)
            if num_picks_to_invest > 0:
                investment_per_stock = current_capital / num_picks_to_invest
            else:
                investment_per_stock = 0

            for code in picks:
                if code in open_positions: continue # 避免重复买入
                if investment_per_stock <= 0: continue # No capital to invest

                stock_data = all_data.get(code)
                if stock_data is None: continue

                # Get the first 5min bar's open price for the current_date
                current_day_data = stock_data[stock_data.index.date == current_date.date()]
                if current_day_data.empty: continue

                buy_price_row = current_day_data.iloc[0] # First 5min bar of the day
                buy_price = buy_price_row["open"]
                buy_datetime = buy_price_row.name # The actual datetime of the first bar

                if pd.isna(buy_price) or buy_price == 0: continue
                
                shares = investment_per_stock / buy_price
                
                # Deduct investment from current capital
                current_capital -= investment_per_stock

                open_positions[code] = {
                    "buy_date": buy_datetime, # Store full datetime
                    "buy_price": buy_price,
                    "shares": shares,
                    "initial_investment": investment_per_stock,
                    "hold_days": 0,
                    "take_profit_triggered": False,
                }

        # 3. 每日检查持仓
        positions_to_close = []
        for code, pos in open_positions.items():
            pos["hold_days"] += 1
            stock_data = all_data.get(code)
            if stock_data is None: continue

            # For 5min data, we need to get the closing price of the last 5min bar of the current_date
            current_day_data = stock_data[stock_data.index.date == current_date.date()]
            if current_day_data.empty or pd.isna(current_day_data["close"].iloc[-1]): continue
            
            current_price = current_day_data["close"].iloc[-1] # Last 5min bar's close price
            pct_change = (current_price - pos["buy_price"]) / pos["buy_price"]

            # 止损检查
            if pct_change <= -stop_loss_pct:
                sell_price = current_price
                proceeds = pos["shares"] * sell_price
                profit_loss = proceeds - pos["initial_investment"]
                current_capital += proceeds # Add proceeds back to capital
                trade_logs.append({
                    "select_date": current_date, "buy_date": pos["buy_date"], "sell_date": current_date,
                    "code": code, "buy_price": pos["buy_price"], "sell_price": sell_price,
                    "pct_change": profit_loss / pos["initial_investment"],
                    "absolute_profit": profit_loss, # Store absolute profit
                })
                positions_to_close.append(code)
                continue

            # 止盈检查 (卖一半)
            if not pos["take_profit_triggered"] and pct_change >= take_profit_pct:
                sell_price = current_price
                shares_to_sell = pos["shares"] / 2
                proceeds = shares_to_sell * sell_price
                
                profit_loss = proceeds - (pos["initial_investment"] / 2)
                current_capital += proceeds # Add proceeds back to capital

                trade_logs.append({
                    "select_date": current_date, "buy_date": pos["buy_date"], "sell_date": current_date,
                    "code": code, "buy_price": pos["buy_price"], "sell_price": sell_price,
                    "pct_change": profit_loss / (pos["initial_investment"] / 2),
                    "absolute_profit": profit_loss, # Store absolute profit
                })
                pos["shares"] /= 2
                pos["initial_investment"] /= 2
                pos["take_profit_triggered"] = True

            # 持有期结束 (sell on the open of the next day)
            if pos["hold_days"] >= hold_period:
                sell_date_idx = i + 1
                if sell_date_idx < len(trade_dates):
                    sell_date = trade_dates[sell_date_idx]
                    
                    # Get the open price of the first 5min bar on the sell_date
                    next_day_data = stock_data[stock_data.index.date == sell_date.date()]
                    if not next_day_data.empty and not pd.isna(next_day_data["open"].iloc[0]):
                        sell_price = next_day_data["open"].iloc[0]
                        sell_datetime = next_day_data.iloc[0].name # The actual datetime of the first bar
                        proceeds = pos["shares"] * sell_price
                        profit_loss = proceeds - pos["initial_investment"]
                        current_capital += proceeds # Add proceeds back to capital
                        trade_logs.append({
                            "select_date": current_date, "buy_date": pos["buy_date"], "sell_date": sell_datetime,
                            "code": code, "buy_price": pos["buy_price"], "sell_price": sell_price,
                            "pct_change": profit_loss / pos["initial_investment"],
                            "absolute_profit": profit_loss, # Store absolute profit
                        })
                        positions_to_close.append(code)

        for code in positions_to_close:
            del open_positions[code]
        
        # At the end of each day, calculate the value of open positions
        current_open_positions_value = 0
        for code, pos in open_positions.items():
            stock_data_today = all_data.get(code)
            if stock_data_today is None: continue
            
            # Get the closing price of the last 5min bar for the current_date
            price_row_today = stock_data_today[stock_data_today.index.date == current_date.date()]
            if price_row_today.empty or pd.isna(price_row_today["close"].iloc[-1]): continue
            current_open_positions_value += pos["shares"] * price_row_today["close"].iloc[-1]

        # The total portfolio value for the day is current_capital (cash) + current_open_positions_value
        # We need to store the *change* in total portfolio value from the initial capital for plotting.
        total_portfolio_value_eod = current_capital + current_open_positions_value
        daily_capital_changes.append({"date": current_date, "total_value": total_portfolio_value_eod})

        # Check for early termination based on total portfolio value
        if total_portfolio_value_eod <= 0:
            logger.info(f"Total portfolio value for {selector_alias} dropped to zero or below on {current_date.strftime('%Y-%m-%d')}. Terminating backtest early.")
            break

    # Convert daily_capital_changes to DataFrame for plotting
    daily_capital_df = pd.DataFrame(daily_capital_changes)
    if not daily_capital_df.empty:
        daily_capital_df = daily_capital_df.set_index("date")
        daily_capital_df["cumulative_profit"] = daily_capital_df["total_value"] - initial_capital
    else:
        daily_capital_df = pd.DataFrame(columns=["date", "total_value", "cumulative_profit"])

    return pd.DataFrame(trade_logs), daily_capital_df


def main():
    p = argparse.ArgumentParser(description="Backtest selectors")
    p.add_argument("--data-dir", default="./data_5min", help="K-line data directory")
    p.add_argument("--config", default="./configs.json", help="Selector config file")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--initial-capital", type=float, default=50000, help="Initial capital for backtesting")
    p.add_argument("--hold", type=int, default=4, help="Holding period in days")
    p.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss percentage (e.g., 0.05 for 5%)")
    p.add_argument("--take-profit", type=float, default=0.05, help="Take profit percentage (e.g., 0.05 for 5%)")
    p.add_argument("--volume-ratio", type=float, default=5.0, help="Volume ratio threshold for opening period (9:30-9:35)")
    args = p.parse_args()

    # --- 加载数据 ---
    data_dir = Path(args.data_dir)
    codes = [f.stem for f in data_dir.glob("*.csv")]
    # Use a modified load_data for 5min data
    all_data = load_data_5min(data_dir, codes)
    if not all_data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    # --- 加载策略 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 执行回测 ---
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    
    all_results = {}
    all_daily_capital_dfs = {} # To store daily capital changes for each alias

    for cfg in selector_cfgs:
        if not cfg.get("activate", True):
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("加载 %s 失败: %s", cfg.get("alias", "N/A"), e)
            continue

        results_df, daily_capital_df = run_backtest(
            start_date, end_date, alias, selector, all_data, args.hold,
            args.stop_loss, args.take_profit,
            initial_capital=args.initial_capital,
            volume_ratio_threshold=args.volume_ratio,
        )

        # --- 打印统计结果 ---
        if results_df.empty:
            logger.info(f"\n============== 回测结果 [{alias}] ==============")
            logger.info("在指定时间段内无任何交易")
            continue

        total_trades = len(results_df)
        win_trades = (results_df["pct_change"] > 0).sum()
        loss_trades = total_trades - win_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        total_return = results_df["pct_change"].sum()
        avg_return = results_df["pct_change"].mean()
        avg_win_return = results_df[results_df["pct_change"] > 0]["pct_change"].mean()
        avg_loss_return = results_df[results_df["pct_change"] <= 0]["pct_change"].mean()

        logger.info(f"\n============== 回测结果 [{alias}] ==============")
        logger.info(f"回测时段: {args.start} to {args.end}")
        logger.info(f"最大持仓周期: {args.hold} 天")
        logger.info(f"止损阈值: {args.stop_loss:.2%}")
        logger.info(f"止盈阈值: {args.take_profit:.2%}")
        logger.info(f"总交易次数: {total_trades}")
        logger.info(f"胜率: {win_rate:.2%}")
        logger.info(f"平均收益率: {avg_return:.2%}")
        logger.info(f"平均盈利收益率: {avg_win_return:.2%}")
        logger.info(f"平均亏损收益率: {avg_loss_return:.2%}")
        logger.info(f"总收益率 (等权重): {total_return:.2%}")
        
        all_results[alias] = results_df
        all_daily_capital_dfs[alias] = daily_capital_df # Store daily capital changes

    # --- 绘制收益曲线 ---
    if not all_daily_capital_dfs: # Check all_daily_capital_dfs instead of all_results
        return
        
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(15, 8))

    for alias, daily_capital_df in all_daily_capital_dfs.items(): # Iterate over daily_capital_dfs
        if daily_capital_df.empty:
            continue
        
        # Plot cumulative profit
        daily_capital_df["cumulative_profit"].plot(ax=ax, label=alias)

    ax.set_title("策略累计盈利变化曲线", fontsize=16) # Changed title
    ax.set_xlabel("日期")
    ax.set_ylabel("每天的累计盈利变化") # Changed y-axis label
    ax.legend()
    ax.grid(True)

    chart_filename = f"backtest_results_hold_{args.hold}_sl_{args.stop_loss}_tp_{args.take_profit}.png"
    chart_path = Path("results") / chart_filename
    chart_path.parent.mkdir(exist_ok=True)
    fig.savefig(chart_path, dpi=300)
    logger.info(f"\n收益曲线图已保存至: {chart_path}")


if __name__ == "__main__":
    main()
