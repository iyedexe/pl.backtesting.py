#!/usr/bin/env bash
# Fetch the raw public data sources used to build research/data/vendored/.
# Usage: scripts/fetch_sources.sh <target-dir>
# Then:  uv run python scripts/build_vendored_data.py --src <target-dir>
set -euo pipefail
DEST=${1:?usage: fetch_sources.sh <target-dir>}
mkdir -p "$DEST" && cd "$DEST"

git clone --depth 1 https://github.com/datasets/exchange-rates
git clone --depth 1 https://github.com/datasets/oil-prices
git clone --depth 1 https://github.com/datasets/natural-gas

git clone --depth 1 --filter=blob:none --sparse https://github.com/coinmetrics-io/data coinmetrics
git -C coinmetrics sparse-checkout set --no-cone '/LICENSE*' '/README.md' \
    '/csv/btc.csv' '/csv/eth.csv' '/csv/ltc.csv' '/csv/bch.csv' '/csv/xrp.csv' \
    '/csv/ada.csv' '/csv/doge.csv' '/csv/sol.csv' '/csv/dot.csv' '/csv/xmr.csv' \
    '/csv/paxg.csv' '/csv/link.csv' '/csv/etc.csv' '/csv/bnb.csv' '/csv/avax.csv'

git clone --depth 1 --filter=blob:none --sparse https://github.com/CNuge/kaggle-code
git -C kaggle-code sparse-checkout set stock_data

git clone --depth 1 --filter=blob:none --sparse https://github.com/robertmartin8/PyPortfolioOpt ppo
git -C ppo sparse-checkout set tests/resources

echo "All sources fetched into $DEST"
