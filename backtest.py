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
):
    """执行回测"""
    trade_logs = []
    all_dates = sorted(
        list(set(d for df in all_data.values() for d in df["date"]))
    )
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]

    for i, current_date in enumerate(tqdm(trade_dates, desc=f"回测 {selector_alias}")):
        # 1. 选股
        picks = selector.select(current_date, all_data)
        if not picks:
            continue

        # 2. 模拟交易
        buy_date_idx = i + 1
        sell_date_idx = i + hold_period
        if sell_date_idx >= len(trade_dates):
            continue  # 超出回测期，无法卖出

        buy_date = trade_dates[buy_date_idx]
        sell_date = trade_dates[sell_date_idx]

        for code in picks:
            stock_data = all_data.get(code)
            if stock_data is None:
                continue

            buy_price_row = stock_data[stock_data["date"] == buy_date]
            sell_price_row = stock_data[stock_data["date"] == sell_date]

            if buy_price_row.empty or sell_price_row.empty:
                continue

            buy_price = buy_price_row["open"].iloc[0]
            sell_price = sell_price_row["open"].iloc[0]

            if pd.isna(buy_price) or pd.isna(sell_price) or buy_price == 0:
                continue

            pct_change = (sell_price - buy_price) / buy_price
            trade_logs.append(
                {
                    "select_date": current_date,
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "code": code,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "pct_change": pct_change,
                }
            )
    return pd.DataFrame(trade_logs)


def main():
    p = argparse.ArgumentParser(description="Backtest selectors")
    p.add_argument("--data-dir", default="./data_day", help="K-line data directory")
    p.add_argument("--config", default="./configs.json", help="Selector config file")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--hold", type=int, default=5, help="Holding period in days")
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

    for cfg in selector_cfgs:
        if not cfg.get("activate", True):
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("加载 %s 失败: %s", cfg.get("alias", "N/A"), e)
            continue

        results_df = run_backtest(
            start_date, end_date, alias, selector, all_data, args.hold
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
        logger.info(f"持仓周期: {args.hold} 天")
        logger.info(f"总交易次数: {total_trades}")
        logger.info(f"胜率: {win_rate:.2%}")
        logger.info(f"平均收益率: {avg_return:.2%}")
        logger.info(f"平均盈利收益率: {avg_win_return:.2%}")
        logger.info(f"平均亏损收益率: {avg_loss_return:.2%}")
        logger.info(f"总收益率 (等权重): {total_return:.2%}")
        
        all_results[alias] = results_df

    # --- 绘制收益曲线 ---
    if not all_results:
        return
        
    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(15, 8))

    for alias, results_df in all_results.items():
        if results_df.empty:
            continue
        # 按卖出日期分组，计算每日总收益率
        daily_return = results_df.groupby("sell_date")["pct_change"].sum() / len(codes)
        # 计算累计收益率
        cumulative_return = (1 + daily_return).cumprod() - 1
        cumulative_return.plot(ax=ax, label=alias)

    ax.set_title("策略累计收益率曲线", fontsize=16)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计收益率")
    ax.legend()
    ax.grid(True)

    chart_path = "backtest_results.png"
    fig.savefig(chart_path, dpi=300)
    logger.info(f"\n收益曲线图已保存至: {chart_path}")


if __name__ == "__main__":
    main()
