# Top-of-Book Collector Runbook (Phase B)

How to run `src/data/collect_top_of_book.py` for a multi-day cross-venue
best-bid/ask collection on the DigitalOcean droplet (`ssh crypto-trader`), then
pull the data back for the executable-gap analysis. §0 records the as-built
deployment; the rest reproduces it from scratch.

**Public market data only** — no API keys, no orders. The collector subscribes to
Kraken Futures (`ticker` feed) and OKX (`bbo-tbt` channel) public websockets.

---

## 0. Deployed layout (as built)

| Thing | Value |
| ----- | ----- |
| Host | `ssh crypto-trader` → <DROPLET_IP> (`alphapulse-collector`, Ubuntu 24.04) |
| Install path | `/opt/alphapulse-rl-btc` |
| Service user | `alphapulse` (system user, `nologin` shell, owns the install path) |
| venv | `/opt/alphapulse-rl-btc/.venv` (~15 MB) |
| Output | `/opt/alphapulse-rl-btc/data/tob/` |
| Unit | `alphapulse-tob.service` |

## 1. Install (on the droplet)

Ubuntu 24.04 ships `python3` without `ensurepip`/`pip`, so `python3 -m venv` fails
even though `python3 -m venv --help` succeeds. Install `python3-venv` first:

```bash
apt-get update && apt-get install -y python3-venv
useradd --system --create-home --home-dir /opt/alphapulse-rl-btc \
        --shell /usr/sbin/nologin alphapulse
```

Sync the code from the Mac. **Anchor the `data/` exclude with a leading slash** —
an unanchored `--exclude 'data/'` also matches `src/data/` and silently drops the
collector itself:

```bash
rsync -avz --no-owner --no-group \
  --exclude '/.venv/' --exclude '/data/' --exclude '/models/' --exclude '/runs/' \
  --exclude '/.git/' --exclude '__pycache__/' --exclude '*.parquet' \
  --exclude '.DS_Store' --exclude '/.pytest_cache/' --exclude '/.claude/' \
  ./ crypto-trader:/opt/alphapulse-rl-btc/
ssh crypto-trader 'chown -R alphapulse:alphapulse /opt/alphapulse-rl-btc'
```

Build the venv. `websockets` is the collector's **only** third-party dependency —
everything else it imports is stdlib, so do NOT install `requirements.txt` (pandas)
or `requirements-train.txt` (torch) here. Analysis runs on the Mac, not the droplet:

```bash
ssh crypto-trader
sudo -u alphapulse -H python3 -m venv /opt/alphapulse-rl-btc/.venv
sudo -u alphapulse -H /opt/alphapulse-rl-btc/.venv/bin/pip install "websockets>=12,<16"
cd /opt/alphapulse-rl-btc && sudo -u alphapulse -H .venv/bin/python \
  -m src.data.collect_top_of_book --help      # import check
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

Smoke-test before committing to a multi-day run:

```bash
cd /opt/alphapulse-rl-btc
sudo -u alphapulse -H mkdir -p /tmp/tob_smoke
sudo -u alphapulse -H .venv/bin/python -m src.data.collect_top_of_book \
  --symbols ARB,NEAR,UNI,BTC --duration 60 --out /tmp/tob_smoke/
# expect: "Flushed N quote rows" every 5s, one `connected` event per venue,
# zero `disconnect` rows in tob_events_*.csv
rm -rf /tmp/tob_smoke
```

### systemd (the deployed method — survives reboots/logouts)

`/etc/systemd/system/alphapulse-tob.service`:

```ini
[Unit]
Description=AlphaPulse cross-venue top-of-book collector (Phase B)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=alphapulse
Group=alphapulse
WorkingDirectory=/opt/alphapulse-rl-btc
ExecStart=/opt/alphapulse-rl-btc/.venv/bin/python -m src.data.collect_top_of_book --symbols ARB,NEAR,UNI,BTC --out /opt/alphapulse-rl-btc/data/tob/
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now alphapulse-tob
```

### Alternative — `nohup` (no root needed)

```bash
cd /opt/alphapulse-rl-btc
sudo -u alphapulse -H mkdir -p data/tob logs
sudo -u alphapulse -H nohup .venv/bin/python -m src.data.collect_top_of_book \
  --symbols ARB,NEAR,UNI,BTC --out data/tob/ \
  > logs/tob_collector.log 2>&1 &
echo $! > logs/tob_collector.pid
```

---

## 3. Check it's alive

```bash
D=/opt/alphapulse-rl-btc/data/tob

# systemd:
systemctl status alphapulse-tob --no-pager
journalctl -u alphapulse-tob -n 50 --no-pager

# nohup:
tail -f /opt/alphapulse-rl-btc/logs/tob_collector.log   # periodic "Flushed N quote rows"

# Data is landing? (files roll daily, UTC)
ls -la $D; du -sh $D
tail -3 $D/tob_$(date -u +%Y%m%d).csv              # latest quotes
cat $D/tob_events_$(date -u +%Y%m%d).csv           # connects/disconnects — an
                                                   # OUTAGE here is data, not a gap
                                                   # in the market; the analyzer
                                                   # excludes outage windows.
# Row growth over ~30s should be ~1.4k (measured ~46 rows/s across 4 assets):
wc -l $D/tob_$(date -u +%Y%m%d).csv; sleep 30; wc -l $D/tob_$(date -u +%Y%m%d).csv

# Per-venue/per-asset coverage — catches a venue silently dropping a symbol,
# which would otherwise read as a "quiet market":
tail -n +2 $D/tob_$(date -u +%Y%m%d).csv | awk -F, '{print $3"  "$4}' | sort | uniq -c
```

If `tob_events_*.csv` shows repeated `disconnect` rows for one venue, that venue is
throttling/blocking — note it; the collector keeps reconnecting with backoff.

---

## 4. Pull the data back (to your laptop)

```bash
# whole directory:
rsync -avz crypto-trader:/opt/alphapulse-rl-btc/data/tob/ ./data/tob/
# or a single day:
scp crypto-trader:/opt/alphapulse-rl-btc/data/tob/tob_20260925.csv ./data/tob/
```

Then analyze locally:

```bash
.venv/bin/python -m src.strategies.dislocation.executable --dir data/tob
```

---

## 5. Stop

```bash
# systemd — stop but keep the unit enabled:
systemctl stop alphapulse-tob
# systemd — stop and prevent restart on boot:
systemctl disable --now alphapulse-tob
# nohup:
kill $(cat /opt/alphapulse-rl-btc/logs/tob_collector.pid)   # graceful; final flush runs
```

`run_collector` installs an asyncio handler for `SIGTERM`/`SIGINT` that sets a stop
event, so the `finally` block runs and flushes whatever is still buffered — a
`systemctl stop` loses at most the rows that arrived in the same event-loop tick, not
the file. Confirm the flush in the journal:

```bash
journalctl -u alphapulse-tob -n 5 --no-pager | grep "final flush complete"
# expect: "Collector stopped; final flush complete (N quote rows, M events)."
```

**If that line is absent, the buffer since the last periodic flush was lost.** Python's
*default* `SIGTERM` action terminates the interpreter immediately — no `finally`, no
flush. The collector ran without the handler until 2026-10-13 and the Phase B shutdown
lost ~200 rows (~2.3 s) that way; harmless against 71.5M rows, but silent. If you ever
run a build predating the handler, stop it with `SIGINT` instead (`kill -INT <pid>`),
which raises `KeyboardInterrupt` and does unwind.

---

## Notes / caveats

- **CSV, not Parquet, on purpose**: a long-lived collector appends line-by-line, so
  a crash never corrupts a whole file. Daily rotation keeps files manageable.
- **Disk (measured, not estimated)**: the 2026-10-01 smoke run produced 2,778 rows
  in 60 s at **64 bytes/row** → ~46 rows/s → ~**255 MB/day** → ~**1.8 GB for 7 days**.
  The droplet has 22 GB free, so ~12× headroom. Re-derive from *observed* growth at
  the 24 h check rather than trusting this figure: OKX `bbo-tbt` is genuinely
  tick-by-tick and a volatile stretch multiplies the rate. If the day-7 projection
  exceeds ~15 GB, drop to 3 symbols or gzip completed daily files rather than
  letting it run blind.
- **The two feeds are wildly asymmetric in update rate.** From the smoke run
  (60 s, rows per venue × asset):

  | Asset | Kraken (`ticker`) | OKX (`bbo-tbt`) |
  | ----- | ----------------: | --------------: |
  | ARB   |  48 |  315 |
  | BTC   |  92 | 1117 |
  | NEAR  |  62 |  600 |
  | UNI   |  60 |  484 |
  | **total** | **262** (~4.4/s) | **2516** (~42/s) |

  Kraken's `ticker` feed delivers roughly **1 update/s/asset** against OKX's tens.
  Cross-venue alignment resolution is therefore bounded by the **Kraken** side at
  ~1 s, not by OKX. That is still a large improvement on Phase A's 5-minute candles
  (the reason Phase B exists), but any claimed dislocation shorter than ~1 s is
  below what this dataset can resolve — the analyzer must not read sub-second
  excursions as real.
- **Deployed via systemd on the droplet** (see §0 for the as-built layout). Always
  do a `--duration 60` smoke run after re-syncing code, before restarting the
  long-lived service.
