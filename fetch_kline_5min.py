import argparse
import logging
import sys
from pathlib import Path
import akshare as ak
import pandas as pd
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fetch_kline_5min")

def fetch_and_save_5min_data(codes, data_dir="data_5min", start_date=None, end_date=None):
    """Fetch 5-minute k-line data for given stock codes and save to CSV."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    for code in tqdm(codes, desc="Fetching 5-min K-line data"):
        try:
            # Fetch 5-minute data from akshare
            stock_zh_a_hist_min_df = ak.stock_zh_a_hist_min_em(symbol=code, period='5', adjust='qfq', start_date=start_date, end_date=end_date)
            
            if stock_zh_a_hist_min_df.empty:
                logger.warning(f"No 5-minute data found for stock code: {code}")
                continue

            # Save to CSV
            file_path = data_dir / f"{code}.csv"
            stock_zh_a_hist_min_df.to_csv(file_path, index=True)
            logger.info(f"Successfully fetched and saved 5-minute data for {code}")
            
            # Add a small delay to avoid being banned by the API
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to fetch 5-minute data for stock code {code}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fetch 5-minute K-line data using akshare.")
    parser.add_argument(
        "--stocks-file",
        default="target_stocks.txt",
        help="File containing a comma-separated list of stock codes.",
    )
    parser.add_argument(
        "--data-dir",
        default="data_5min",
        help="Directory to save the 5-minute K-line data.",
    )
    parser.add_argument(
        "--start-date",
        required=False,
        help="Start date for fetching data (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        required=False,
        help="End date for fetching data (YYYYMMDD)",
    )
    args = parser.parse_args()

    # Read stock codes from file
    try:
        with open(args.stocks_file, "r") as f:
            codes = f.read().strip().split(',')
    except FileNotFoundError:
        logger.error(f"Error: The file '{args.stocks_file}' was not found.")
        sys.exit(1)

    if not codes:
        logger.error("No stock codes found in the specified file.")
        sys.exit(1)

    # Fetch and save data
    fetch_and_save_5min_data(codes, args.data_dir, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
