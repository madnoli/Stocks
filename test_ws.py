# td_replay_diagnostics.py
import os
import sys
import time
import socket
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime as dt
import requests
import pandas as pd

from truedata_ws.websocket.TD import TD

HOST = "replay.truedata.in"     # replay host for backtesting
PORT = "8084"                 # default realtime port for replay
USER = "tdwsp751"
PASS = "raj@751"
SYMBOL = "SBIN"                 # test underlying for chain
EXPIRY = dt(2025, 10, 28)       # test expiry
CHAIN_LEN = 40

REST_CHAIN_URL = "https://api.truedata.in/getOptionChain"

def build_logger():
    logger = logging.getLogger("td_replay")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    fh = RotatingFileHandler("td_replay.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

log = build_logger()

def probe_network(host: str, port: int):
    # DNS resolution
    t0 = time.time()
    try:
        addr = socket.gethostbyname(host)
        log.info(f"DNS resolved | {host} -> {addr} | dt_ms={int((time.time()-t0)*1000)}")
    except Exception as e:
        log.error(f"DNS failure | host={host} | err={e}")
        return False

    # TCP reachability
    t1 = time.time()
    try:
        with socket.create_connection((host, port), timeout=5) as s:
            log.info(f"TCP ok | {host}:{port} | dt_ms={int((time.time()-t1)*1000)}")
            return True
    except Exception as e:
        log.error(f"TCP connect failed | {host}:{port} | err={e}")
        return False

def rest_sanity(user: str, pwd: str, symbol: str, expiry_str: str):
    try:
        params = {"user": user, "password": pwd, "symbol": symbol, "expiry": expiry_str}
        r = requests.get(REST_CHAIN_URL, params=params, timeout=10)
        r.raise_for_status()
        # Log just status and size to avoid printing credentials or large payloads
        log.info(f"REST ok | status={r.status_code} | bytes={len(r.text)}")
        return True
    except Exception as e:
        log.error(f"REST failed | err={e}")
        return False

def main():
    log.info(f"Replay connect start | host={HOST} port={PORT} symbol={SYMBOL} expiry={EXPIRY.date()}")

    # Environment info
    try:
        import truedata_ws
        log.info(f"truedata_ws version: {getattr(truedata_ws, '__version__', 'unknown')}")
    except Exception as e:
        log.warning(f"Could not read truedata_ws version | err={e}")

    # Network probes
    net_ok = probe_network(HOST, PORT)

    # REST sanity check (auth + backend reachability)
    rest_ok = rest_sanity(USER, PASS, SYMBOL, EXPIRY.strftime("%Y%m%d"))

    # Initialize TD client with verbose internal logs
    t2 = time.time()
    try:
        td = TD(
            USER, PASS,
            url=HOST, live_port=PORT,
            log_level=logging.DEBUG,
            log_format="(%(asctime)s) %(levelname)s :: %(message)s (PID:%(process)d Thread:%(thread)d)"
        )
        log.info(f"TD init ok | dt_ms={int((time.time()-t2)*1000)}")
    except Exception as e:
        log.error(f"TD init failed | err={e}")
        sys.exit(1)

    # Subscribe to replayed option chain
    try:
        log.info(f"Subscribing option chain | {SYMBOL} {EXPIRY.date()} len={CHAIN_LEN}")
        chain = td.start_option_chain(SYMBOL, EXPIRY, chain_length=CHAIN_LEN, bid_ask=True)
        log.info("Option chain subscribed")
    except Exception as e:
        log.error(f"start_option_chain failed | err={e}")
        try:
            td.disconnect()
        except Exception:
            pass
        sys.exit(1)

    # Snapshot activity loop: track row count changes and a few key columns
    last_rows = -1
    last_non_empty = 0
    try:
        while True:
            df = chain.get_option_chain()
            nrows = len(df) if df is not None else 0

            # Count non-empty LTP/OI to check data liveliness
            non_empty_ltp = int(df["ltp"].notna().sum()) if df is not None and "ltp" in df.columns else 0
            non_empty_oi = int(df["oi"].notna().sum()) if df is not None and "oi" in df.columns else 0

            # Log when row count or non-empty metrics change
            if nrows != last_rows or non_empty_ltp != last_non_empty:
                log.info(f"Chain snapshot | rows={nrows} ltp_nonempty={non_empty_ltp} oi_nonempty={non_empty_oi}")
                last_rows = nrows
                last_non_empty = non_empty_ltp

            # If nothing is populated, remind about replay window/expiry
            if nrows and non_empty_ltp == 0 and non_empty_oi == 0:
                log.debug("Snapshot with empty LTP/OI; confirm replay window/time and chosen expiry/chain_len")

            time.sleep(1)
    except KeyboardInterrupt:
        log.warning("Ctrl+C received, stopping...")
    finally:
        try:
            chain.stop_option_chain()
            log.info("Stopped option chain")
        except Exception as e:
            log.error(f"stop_option_chain error | {e}")
        try:
            td.disconnect()
            log.info("Disconnected TD")
        except Exception as e:
            log.error(f"disconnect error | {e}")

if __name__ == "__main__":
    main()
