# Top-of-Book Collector Runbook (Phase B)

How to run `src/data/collect_top_of_book.py` for a multi-day cross-venue
best-bid/ask collection on the DigitalOcean droplet (`ssh crypto-trader`), then
pull the data back for the executable-gap analysis.

**Public market data only** — no API keys, no orders. The collector subscribes to
Kraken Futures (`ticker` feed) and OKX (`bbo-tbt` channel) public websockets.

---

## 1. Install (on the droplet)

```bash
ssh crypto-trader
cd ~/alphapulse-rl-btc          # or: git clone <repo> && cd alphapulse-rl-btc
git pull
python3 -m venv .venv           # if not already present
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install "websockets>=12,<16"   # the only extra dep for the collector
```

### Clock (do this — cross-venue alignment depends on it)

The analyzer aligns venues on **one clock**: the collector's receive time
(`ts_recv`). Make sure the droplet's clock is NTP-synced so `ts_recv` is accurate:

```bash
timedatectl                     # want: "System clock synchronized: yes", NTP active
sudo timedatectl set-ntp true   # if not already
```

The collector still records each venue's own `ts_exchange` too, but analysis keys
off `ts_recv` so a single machine clock removes cross-venue clock skew.

---

## 2. Launch (long-lived, ~5–7 days)

### Option A — `nohup` (simplest)

```bash
cd ~/alphapulse-rl-btc
mkdir -p data/tob logs
nohup .venv/bin/python -m src.data.collect_top_of_book \
  --symbols ARB,NEAR,UNI,BTC --out data/tob/ \
  > logs/tob_collector.log 2>&1 &
echo $! > logs/tob_collector.pid
```

### Option B — `systemd` (survives reboots/logouts)

Create `/etc/systemd/system/tob-collector.service`:

```ini
[Unit]
Description=Cross-venue top-of-book collector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/alphapulse-rl-btc
ExecStart=/home/YOUR_USER/alphapulse-rl-btc/.venv/bin/python -m src.data.collect_top_of_book --symbols ARB,NEAR,UNI,BTC --out data/tob/
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tob-collector
```

---

## 3. Check it's alive

```bash
# nohup:
tail -f logs/tob_collector.log            # expect periodic "Flushed N quote rows"
ps -p $(cat logs/tob_collector.pid)

# systemd:
systemctl status tob-collector
journalctl -u tob-collector -f

# Data is landing? (files roll daily, UTC)
ls -la data/tob/
tail -3 data/tob/tob_$(date -u +%Y%m%d).csv       # latest quotes
cat data/tob/tob_events_$(date -u +%Y%m%d).csv     # connects/disconnects — an
                                                   # OUTAGE here is data, not a gap
                                                   # in the market; the analyzer
                                                   # excludes outage windows.
# Row growth over ~30s should be thousands (BTC/liquid alts update fast):
wc -l data/tob/tob_$(date -u +%Y%m%d).csv; sleep 30; wc -l data/tob/tob_$(date -u +%Y%m%d).csv
```

If `tob_events_*.csv` shows repeated `disconnect` rows for one venue, that venue is
throttling/blocking — note it; the collector keeps reconnecting with backoff.

---

## 4. Pull the data back (to your laptop)

```bash
# whole directory:
rsync -avz crypto-trader:~/alphapulse-rl-btc/data/tob/ ./data/tob/
# or a single day:
scp crypto-trader:~/alphapulse-rl-btc/data/tob/tob_20260925.csv ./data/tob/
```

Then analyze locally:

```bash
.venv/bin/python -m src.strategies.dislocation.executable --dir data/tob
```

---

## 5. Stop

```bash
# nohup:
kill $(cat logs/tob_collector.pid)        # graceful; final flush runs on shutdown
# systemd:
sudo systemctl disable --now tob-collector
```

---

## Notes / caveats

- **CSV, not Parquet, on purpose**: a long-lived collector appends line-by-line, so
  a crash never corrupts a whole file. Daily rotation keeps files manageable.
- **Disk**: liquid symbols emit many updates/sec. Budget ~roughly a few hundred MB
  per symbol per day at TOB tick rate; `du -sh data/tob/` periodically.
- **Deploy is manual** — this runbook is for you to run; the assistant does not
  deploy. Do a `--duration 120` local smoke run first to confirm connectivity and
  schema parsing before committing a multi-day run.
