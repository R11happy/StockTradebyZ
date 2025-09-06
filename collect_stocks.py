from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from select_stock import instantiate_selector, load_config, load_data

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_stocks")


def main():
    p = argparse.ArgumentParser(description="Collect all selected stocks over a period")
    p.add_argument("--data-dir", default="./data_day", help="K-line data directory for selectors")
    p.add_argument("--config", default="./configs.json", help="Selector config file")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--out", default="./target_stocks.txt", help="Output file for collected stock codes")
    args = p.parse_args()

    # --- 加载日线数据 (选股逻辑需要) ---
    data_dir = Path(args.data_dir)
    codes = [f.stem for f in data_dir.glob("*.csv")]
    all_data = load_data(data_dir, codes)
    if not all_data:
        logger.error("未能加载任何日线行情数据")
        sys.exit(1)

    # --- 加载策略 ---
    selector_cfgs = load_config(Path(args.config))
    selectors = []
    for cfg in selector_cfgs:
        if not cfg.get("activate", True):
            continue
        try:
            alias, selector = instantiate_selector(cfg)
            selectors.append(selector)
        except Exception as e:
            logger.error("加载 %s 失败: %s", cfg.get("alias", "N/A"), e)
            continue
    
    if not selectors:
        logger.error("未能加载任何选股策略")
        sys.exit(1)

    # --- 确定回测日期范围 ---
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    all_dates = sorted(list(set(d for df in all_data.values() for d in df["date"])))
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]

    # --- 收集所有选中的股票 ---
    all_picks = set()
    for current_date in tqdm(trade_dates, desc="Collecting stocks"):
        for selector in selectors:
            picks = selector.select(current_date, all_data)
            if picks:
                all_picks.update(picks)

    # --- 保存结果 ---
    out_path = Path(args.out)
    sorted_picks = sorted(list(all_picks))
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(sorted_picks))

    logger.info(f"\n在 {args.start} 到 {args.end} 期间共选中 {len(sorted_picks)} 只不重复的股票.")
    logger.info(f"股票列表已保存至: {out_path}")


if __name__ == "__main__":
    main()
