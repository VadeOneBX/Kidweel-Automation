#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/logs

# 1. Start the Cron scheduler in the background
/usr/local/bin/supercronic /app/docker/crontab >> /app/logs/cron.log 2>&1 &

# 2. Start the Live Trader Daemon (Listens to Redis queue for trade signals)
echo "🚀 Starting QOPS Live Trader Daemon..."
exec /usr/local/bin/python -m qops.engine.live_trader