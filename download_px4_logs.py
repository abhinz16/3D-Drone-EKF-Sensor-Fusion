#!/usr/bin/env python3
"""Download public PX4 Flight Review logs.

Examples:
    python download_px4_logs.py
    python download_px4_logs.py --max-num 5 --rating Good --mav-type Quadrotor
    python download_px4_logs.py --print

The default is intentionally small because the PX4 Flight Review server is a
shared public resource. Use --max-num -1 only if you know what you are doing.
"""

import argparse
import json
import os
import sys

from src.px4_downloader import (
    DEFAULT_DELAY_SECONDS,
    WARN_THRESHOLD,
    download_public_logs,
    fetch_public_log_index,
    filter_logs,
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Download public logs from PX4 Flight Review.")
    parser.add_argument("--max-num", "-n", type=int, default=10,
                        help="Maximum logs to download. Use -1 for all matches, with confirmation.")
    parser.add_argument("-d", "--download-folder", default="data/downloaded",
                        help="Folder where .ulg files are stored.")
    parser.add_argument("--print", action="store_true", dest="print_entries",
                        help="Print matching database entries instead of downloading.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--mav-type", nargs="+", default=["Quadrotor"],
                        help="Filter by MAV type, for example Quadrotor.")
    parser.add_argument("--rating", nargs="+", default=["Good"],
                        help="Filter by rating, for example Good.")
    parser.add_argument("--source", default=None)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                        help="Delay between downloads to respect rate limits.")
    parser.add_argument("--yes", "-y", action="store_true", default=False,
                        help="Skip confirmation for large downloads.")
    return parser.parse_args()


def confirm_large_download(n_files):
    print("\n" + "=" * 60)
    print(f"WARNING: You are about to download {n_files} files.")
    print("=" * 60)
    print("PX4 Flight Review is a shared public service. Large downloads should")
    print("be done slowly and only when you actually need the logs.")
    response = input("Continue with download? [y/N]: ")
    return response.lower() in ["y", "yes"]


def main():
    args = get_arguments()

    entries = fetch_public_log_index()
    entries = filter_logs(entries, mav_type=args.mav_type, rating=args.rating, source=args.source)

    print(f"Matched public logs: {len(entries)}")
    if args.print_entries:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return

    if args.max_num == -1:
        n_to_download = len(entries)
    else:
        n_to_download = min(len(entries), args.max_num)

    if n_to_download > WARN_THRESHOLD and not args.yes:
        if not confirm_large_download(n_to_download):
            print("Download cancelled.")
            sys.exit(0)

    downloaded = download_public_logs(
        download_folder=args.download_folder,
        max_num=args.max_num,
        mav_type=args.mav_type,
        rating=args.rating,
        source=args.source,
        overwrite=args.overwrite,
        delay=args.delay,
    )

    print("\nDownload complete")
    print(f"Files available in: {os.path.abspath(args.download_folder)}")
    print(f"Downloaded or found existing: {len(downloaded)}")


if __name__ == "__main__":
    main()
