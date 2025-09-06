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

from select_stock import instantiate_selector, load_config, load_data

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backtest")


def run_backtest(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    selector_alias: str,
    selector,
    all_data: dict[str, pd.DataFrame],
    hold_period: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    initial_capital: float, # Added parameter
):
    """执行回测"""
    trade_logs = []
    all_dates = sorted(list(set(d for df in all_data.values() for d in df["date"])))
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]
    
    # 用于跟踪持仓信息
    open_positions = {}
    
    # 用于跟踪每日的资金变化
    daily_capital_changes = []
    current_capital = initial_capital

    for i, current_date in enumerate(tqdm(trade_dates, desc=f"回测 {selector_alias}")):
        # 1. 选股
        picks = selector.select(current_date, all_data)
        
        # 2. 模拟交易 - 买入
        if picks:
            # Find the next trading day for buying
            buy_date = None
            for j in range(i + 1, len(trade_dates)):
                # Ensure we are not buying on a weekend
                if trade_dates[j].weekday() < 5: # Monday is 0, Friday is 4
                    buy_date = trade_dates[j]
                    break
            
            if buy_date:
                
                # Calculate investment per stock based on available capital and number of picks
                # Ensure we don't invest more than available capital
                num_picks_to_invest = len(picks)
                if num_picks_to_invest > 0:
                    # Distribute available capital among selected stocks
                    investment_per_stock = current_capital / num_picks_to_invest
                else:
                    investment_per_stock = 0

                for code in picks:
                    if code in open_positions: continue # 避免重复买入
                    if investment_per_stock <= 0: continue # No capital to invest

                    stock_data = all_data.get(code)
                    if stock_data is None: continue

                    buy_price_row = stock_data[stock_data["date"] == buy_date]
                    if buy_price_row.empty: continue

                    buy_price = buy_price_row["open"].iloc[0]
                    if pd.isna(buy_price) or buy_price == 0: continue
                    
                    shares = investment_per_stock / buy_price
                    
                    # Deduct investment from current capital
                    current_capital -= investment_per_stock

                    open_positions[code] = {
                        "buy_date": buy_date,
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

            price_row = stock_data[stock_data["date"] == current_date]
            if price_row.empty or pd.isna(price_row["close"].iloc[0]): continue
            
            current_price = price_row["close"].iloc[0]
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

            # 持有期结束
            if pos["hold_days"] >= hold_period:
                sell_date_idx = i + 1
                if sell_date_idx < len(trade_dates):
                    sell_date = trade_dates[sell_date_idx]
                    sell_price_row = stock_data[stock_data["date"] == sell_date]
                    if not sell_price_row.empty and not pd.isna(sell_price_row["open"].iloc[0]):
                        sell_price = sell_price_row["open"].iloc[0]
                        proceeds = pos["shares"] * sell_price
                        profit_loss = proceeds - pos["initial_investment"]
                        current_capital += proceeds # Add proceeds back to capital
                        trade_logs.append({
                            "select_date": current_date, "buy_date": pos["buy_date"], "sell_date": sell_date,
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
            price_row_today = stock_data_today[stock_data_today["date"] == current_date]
            if price_row_today.empty or pd.isna(price_row_today["close"].iloc[0]): continue
            current_open_positions_value += pos["shares"] * price_row_today["close"].iloc[0]

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
    p.add_argument("--data-dir", default="./data_day", help="K-line data directory")
    p.add_argument("--config", default="./configs.json", help="Selector config file")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--initial-capital", type=float, default=50000, help="Initial capital for backtesting") # Added argument
    p.add_argument("--hold", type=int, default=4, help="Holding period in days")
    p.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss percentage (e.g., 0.05 for 5%)")
    p.add_argument("--take-profit", type=float, default=0.05, help="Take profit percentage (e.g., 0.05 for 5%)")
    args = p.parse_args()

    # --- 加载数据 ---
    data_dir = Path(args.data_dir)
    codes = [f.stem for f in data_dir.glob("*.csv")]
    all_data = load_data(data_dir, codes)
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

        results_df, daily_capital_df = run_backtest( # Modified to get two return values
            start_date, end_date, alias, selector, all_data, args.hold,
            args.stop_loss, args.take_profit, initial_capital=args.initial_capital # Pass initial_capital
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
