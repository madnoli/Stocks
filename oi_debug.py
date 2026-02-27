# ======== ENHANCED ANALYSIS WITH CLEAN LOGGING ========
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
    """Enhanced analysis with clean logging"""
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE'

    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    oi_status = 'Normal'
    has_strong_conditions = False

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue

        is_valid, _ = validate_data_quality(df, 20)
        if not is_valid:
            continue

        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)
        
        # Track OI symbols (silent)
        if 'OI' in df.columns and df['OI'].sum() > 100:
            global oi_symbols_found
            oi_symbols_found.add(symbol)
        
        scores = {}
        
        # RSI Analysis
        try:
            rsi_series = calculate_rsi_improved(df)
            if len(rsi_series) >= 2:
                rsi_current = rsi_series.iloc[-1]
                
                if rsi_current > 65:
                    scores['RSI'] = 2.0
                elif rsi_current > 55:
                    scores['RSI'] = 1.0
                elif rsi_current < 35:
                    scores['RSI'] = -2.0
                elif rsi_current < 45:
                    scores['RSI'] = -1.0
                else:
                    scores['RSI'] = 0.0
            else:
                scores['RSI'] = 0.0
        except Exception:
            scores['RSI'] = 0.0
        
        # Momentum Analysis
        try:
            mom = momentum_improved(df)
            if len(mom) >= 2:
                current_mom = mom.iloc[-1]
                if current_mom > 0.01:
                    scores['Momentum'] = 2.0
                elif current_mom > 0.003:
                    scores['Momentum'] = 1.0
                elif current_mom < -0.01:
                    scores['Momentum'] = -2.0
                elif current_mom < -0.003:
                    scores['Momentum'] = -1.0
                else:
                    scores['Momentum'] = 0.0
            else:
                scores['Momentum'] = 0.0
        except Exception:
            scores['Momentum'] = 0.0
        
        # Volume surge analysis
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                
                if current_surge >= 1.5:
                    if price_change > 0.005:
                        scores['VolumeSurge'] = 2.0
                    elif price_change < -0.005:
                        scores['VolumeSurge'] = -2.0
                    else:
                        scores['VolumeSurge'] = 1.0
                elif current_surge >= 1.0:
                    scores['VolumeSurge'] = 1.0 if price_change > 0 else -1.0
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0
        
        # OI analysis
        try:
            oi_z = oi_surge_improved(df)
            if len(oi_z) >= 2:
                oi_surge_current = oi_z.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                
                if oi_surge_current >= 1.5:
                    scores['OISurge'] = 2.0 if price_change > 0 else -2.0
                    oi_status = 'High OI Activity'
                elif oi_surge_current >= 0.8:
                    scores['OISurge'] = 1.0 if price_change > 0 else -1.0
                    oi_status = 'Moderate OI Activity'
                else:
                    scores['OISurge'] = 0.0
            else:
                scores['OISurge'] = 0.0
        except Exception:
            scores['OISurge'] = 0.0
        
        # OI momentum
        try:
            oi_mom = oi_momentum_improved(df)
            if len(oi_mom) >= 2:
                oi_mom_current = oi_mom.iloc[-1]
                if oi_mom_current > 0.05:
                    scores['OIMomentum'] = 2.0
                elif oi_mom_current > 0.02:
                    scores['OIMomentum'] = 1.0
                elif oi_mom_current < -0.02:
                    scores['OIMomentum'] = -1.0
                else:
                    scores['OIMomentum'] = 0.0
            else:
                scores['OIMomentum'] = 0.0
        except Exception:
            scores['OIMomentum'] = 0.0
        
        # Call/Put bias analysis
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
        vol_high = scores.get('VolumeSurge', 0) >= 1.0
        oi_active = abs(scores.get('OISurge', 0)) >= 1.0
        
        if price_up and vol_high and oi_active:
            scores['CallBias'] = 3.0
            scores['PutBias'] = 0.0
            oi_status = 'Call Setup'
            has_strong_conditions = True
        elif not price_up and vol_high and oi_active:
            scores['PutBias'] = -3.0
            scores['CallBias'] = 0.0
            oi_status = 'Put Setup'
            has_strong_conditions = True
        elif price_up and (vol_high or oi_active):
            scores['CallBias'] = 1.0
            scores['PutBias'] = 0.0
        elif not price_up and (vol_high or oi_active):
            scores['PutBias'] = -1.0
            scores['CallBias'] = 0.0
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0
        
        # OI-Volume Confirmation
        if oi_active or vol_high:
            if price_up:
                scores['OIVolConfirm'] = 1.0
            else:
                scores['OIVolConfirm'] = -1.0
        else:
            scores['OIVolConfirm'] = 0.0
        
        # Fill remaining indicators
        remaining_indicators = ['MACD', 'ADX', 'VWAP', 'EMA', 'CMF', 'ADL', 'OBV', 'ATR', 
                               'Bollinger', 'ROC', 'Stochastic', 'CCI', 'MA', 'WWL', 
                               'RelVol', 'VWAPRegime', 'OBVConfirm']
        for indicator in remaining_indicators:
            if indicator not in scores:
                scores[indicator] = 0.0
        
        # Calculate weighted scores
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 3.0 * tf_weight * ind_weight
    
    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'
    
    normalized = (final_score / max_possible) * 100.0
    
    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100
    
    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized)
    
    return signal_text, normalized, oi_status, option_action, alert_priority

# ======== CLEAN DATA FETCHING ========
@retry(
    stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"],
    wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)),
    wait_exponential_max=10000,
    retry_on_exception=lambda e: True
)
def fetch_one_clean(symbol_orig, timeframe_minutes, limiter, hist):
    """Clean fetch function without debug logs"""
    td_symbol = symbol_orig.replace("-EQ", "")
    
    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe_minutes)
    duration = CONFIG["DURATION_MAP"].get(timeframe_minutes)
    
    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        
        if df_raw is None or df_raw.empty:
            return symbol_orig, timeframe_minutes, None
        
        df = normalize_hist_df_clean(df_raw, td_symbol, timeframe_minutes)
        
        # Silent API counter
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        
        return symbol_orig, timeframe_minutes, df
        
    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_clean(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    """CLEAN: Enhanced prefetch with essential info only"""
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    
    global api_calls_done, oi_symbols_found
    with api_calls_lock:
        api_calls_done = 0
    
    oi_symbols_found = set()
    valid_stocks = [s for s in stocks if s]
    
    # CLEAN: Show essential startup info
    console.print(f"📊 Analyzing [cyan]{len(valid_stocks)} symbols[/cyan] across [yellow]5 timeframes[/yellow]")
    
    progress_kwargs = dict(
        total=total_calls,
        desc="🔄 Loading Market Data",
        ncols=80,
        disable=not CONFIG["SHOW_PROGRESS"]
    )
    
    with tqdm(**progress_kwargs) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in valid_stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one_clean, s, tf, sess_limiters[si], tdhist_pool[si]))
            
            for fut in as_completed(futures):
                try:
                    symbol_orig, tf, df = fut.result()
                    if df is not None:
                        stock_multi_data[symbol_orig][tf] = df
                except Exception:
                    pass
                api_bar.update(1)
    
    valid_data = {s: d for s, d in stock_multi_data.items() if len(d) >= 1}
    
    # CLEAN: Essential completion info
    console.print(f"✅ Data loaded: [green]{len(valid_data)} symbols[/green] ready")
    if len(oi_symbols_found) > 0:
        console.print(f"📈 OI data: [yellow]{len(oi_symbols_found)} symbols[/yellow]")
    
    return valid_data

def filter_timeframe_data(symbol, timeframe_data, time_point_aware):
    """Enhanced filtering with clean error handling"""
    filtered_timeframes = {}
    
    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue
        
        try:
            if time_point_aware.tzinfo is None:
                time_point_aware = IST.localize(time_point_aware)
            elif time_point_aware.tzinfo != IST:
                time_point_aware = time_point_aware.astimezone(IST)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert(IST)
            
            try:
                valid_index = df.index.dropna()
                if len(valid_index) != len(df.index):
                    df = df.loc[valid_index]
                
                if not df.empty:
                    mask = df.index <= time_point_aware
                    df_filtered = df.loc[mask]
                    
                    if len(df_filtered) >= CONFIG["MIN_BARS_REQUIRED"]:
                        filtered_timeframes[tf] = df_filtered
                        
            except Exception:
                continue
                
        except Exception:
            continue
    
    return filtered_timeframes

# ======== CLEAN RENDERING ========
def render_signals_clean(now_ts, top_bullish, top_bearish):
    """Clean signal rendering for production"""
    global last_bull_symbols, last_bear_symbols
    
    title = f"🎯 STRONG SIGNALS | {now_ts.strftime('%H:%M')} IST"
    console.rule(title, style="bold yellow")

    # Filter signal categories
    ultra_strong_bulls = [r for r in top_bullish if "ULTRA STRONG" in r['signal']]
    very_strong_bulls = [r for r in top_bullish if "VERY STRONG" in r['signal']]
    strong_bulls = [r for r in top_bullish if "STRONG BUY" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bulls = [r for r in top_bullish if "🟢 BUY" in r['signal']]
    
    ultra_strong_bears = [r for r in top_bearish if "ULTRA STRONG" in r['signal']]
    very_strong_bears = [r for r in top_bearish if "VERY STRONG" in r['signal']]
    strong_bears = [r for r in top_bearish if "STRONG SELL" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bears = [r for r in top_bearish if "🔴 SELL" in r['signal']]

    # Show strong signals
    all_strong = ultra_strong_bulls + very_strong_bulls + strong_bulls + ultra_strong_bears + very_strong_bears + strong_bears
    all_moderate = moderate_bulls + moderate_bears
    
    if all_strong:
        console.print("\n🔥 [bold white on red]STRONG SIGNALS[/bold white on red]")
        
        strong_table = Table(title="💪 STRONG SIGNALS", box=box.DOUBLE_EDGE, header_style="bold white on blue")
        strong_table.add_column("Stock", style="bold white")
        strong_table.add_column("Signal", style="bold yellow")
        strong_table.add_column("Score", style="bold green", justify="right")
        strong_table.add_column("OI Status", style="cyan")
        strong_table.add_column("Action", style="bold red")
        
        for r in all_strong:
            row_style = "bold black on yellow" if r['symbol'] not in (last_bull_symbols | last_bear_symbols) else None
            strong_table.add_row(
                r['symbol'], r['signal'], f"{r['score']:.1f}",
                r.get('oi_status', 'Normal'), r.get('action', 'TRADE'),
                style=row_style
            )
        
        console.print(strong_table)
    
    if all_moderate and len(all_moderate) > 0:
        console.print("\n📊 [bold blue]MODERATE SIGNALS[/bold blue]")
        
        mod_table = Table(title="📈 MODERATE SIGNALS", box=box.SIMPLE, header_style="bold white on green")
        mod_table.add_column("Stock", style="cyan")
        mod_table.add_column("Signal", style="white")
        mod_table.add_column("Score", style="yellow", justify="right")
        mod_table.add_column("Action", style="green")
        
        for r in all_moderate[:10]:
            mod_table.add_row(
                r['symbol'], r['signal'], f"{r['score']:.1f}",
                r.get('action', 'CONSIDER')
            )
        
        console.print(mod_table)

    # Summary
    total_ultra = len(ultra_strong_bulls + ultra_strong_bears)
    total_very = len(very_strong_bulls + very_strong_bears)
    total_strong = len(strong_bulls + strong_bears)
    total_moderate = len(all_moderate)
    
    summary = f"🎯 ULTRA: {total_ultra} | 🔥 VERY STRONG: {total_very} | ⚡ STRONG: {total_strong} | 📊 MODERATE: {total_moderate}"
    console.print(f"\n[bold yellow]{summary}[/bold yellow]")
    console.rule()

    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

def infer_institutional_flow(tf_data):
    """Simplified institutional flow detection"""
    return "Mixed"

def export_to_csv(now_ts, top_bullish, top_bearish, filename):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Time: {now_ts.strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow(["Stock", "Signal", "Score", "Change", "OI Status", "Action"])
        
        for r in top_bullish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        
        for r in top_bearish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        writer.writerow([])

# ======== MAIN BACKTEST FUNCTION - CLEAN ========
def run_backtest_clean(day_str: str, stocks):
    """Clean backtest with essential information only"""
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    console.print(f"📅 [bold cyan]Backtesting {day_str}[/bold cyan] with [yellow]{len(stocks)} symbols[/yellow]")
    
    stock_multi_data = prefetch_clean(stocks)
    
    if len(stock_multi_data) == 0:
        console.print("[red]❌ No valid data found[/red]")
        return

    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_signals_clean.csv"
    
    # Clean old file
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
    except Exception:
        pass

    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    console.print(f"🔍 Analyzing [cyan]{len(checkpoints)}[/cyan] time periods...")

    for i, asof_ts in enumerate(checkpoints):
        if i % 20 == 0:
            console.print(f"⏳ Progress: [cyan]{i+1}/{len(checkpoints)}[/cyan] | Time: [yellow]{asof_ts.strftime('%H:%M')}[/yellow]")
            
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)
            
            if len(filtered_timeframes) < 1:
                continue

            signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(filtered_timeframes, clean_symbol)
            current_scores[clean_symbol] = score
            
            if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)
                
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'oi_status': oi_status, 
                    'flow': flow_tag, 'action': option_action
                })
                performance_metrics[f"{direction}_signals"] += 1

        previous_scores = current_scores.copy()
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

        if top_bullish or top_bearish:
            render_signals_clean(asof_ts, top_bullish, top_bearish)
            export_to_csv(asof_ts, top_bullish, top_bearish, output_filename)

    # Final summary
    total_signals = sum(performance_metrics.values())
    console.print(f"\n📈 [bold green]BACKTEST COMPLETE[/bold green]")
    console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
    if 'bullish_signals' in performance_metrics:
        console.print(f"Bullish: [green]{performance_metrics['bullish_signals']}[/green]")
    if 'bearish_signals' in performance_metrics:
        console.print(f"Bearish: [red]{performance_metrics['bearish_signals']}[/red]")
    console.print(f"Results: [yellow]{output_filename}[/yellow]")

def load_stock_list(file_name):
    """Load stock symbols with clean output"""
    if not os.path.exists(file_name):
        sample_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO",
            "WIPRO", "NESTLEIND", "HCLTECH", "BAJFINANCE", "TITAN", "POWERGRID", "NTPC", "ONGC",
            "TECHM", "DRREDDY", "BAJAJFINSV", "INDUSINDBK", "CIPLA", "COALINDIA", "GRASIM", "BPCL",
            "TATASTEEL", "HINDALCO", "ADANIPORTS", "BRITANNIA", "DIVISLAB", "TATAMOTORS", "HEROMOTOCO",
            "JSWSTEEL", "SHREECEM", "UPL", "APOLLOHOSP", "BAJAJ-AUTO", "EICHERMOT", "SBILIFE"
        ]
        
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                for stock in sample_stocks:
                    f.write(f"{stock}\n")
            console.print(f"📝 Created [yellow]{file_name}[/yellow] with [cyan]{len(sample_stocks)}[/cyan] stocks")
            return sample_stocks
        except Exception:
            return []
    
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        stocks = []
        for line in lines:
            if line and not line.startswith('#'):
                symbol = line.split(',')[0].split('\t')[0].strip().upper()
                if symbol:
                    stocks.append(symbol)
        
        console.print(f"📈 Loaded [cyan]{len(stocks)}[/cyan] symbols from [yellow]{file_name}[/yellow]")
        return stocks
        
    except Exception:
        return []

def print_clean_banner():
    """Clean production banner"""
    console.print("\n" + "="*70, style="bold blue")
    console.print("🎯 [bold cyan]OPTION SIGNAL SCANNER v3.0[/bold cyan] 🎯", justify="center")
    console.print("="*70, style="bold blue")
    console.print(f"🕐 [bold white]{datetime.now(IST).strftime('%H:%M:%S IST')}[/bold white] | Mode: [green]PRODUCTION[/green]")
    console.print("="*70, style="bold blue")

# ======== MAIN EXECUTION - CLEAN ========
if __name__ == "__main__":
    print_clean_banner()
    
    try:
        parser = argparse.ArgumentParser(description="Clean Option Signal Scanner")
        parser.add_argument("--backtest-date", help="Backtest date (YYYY-MM-DD)")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="Stock symbols file")
        parser.add_argument("--live", action="store_true", help="Live market mode")
        parser.add_argument("--test-oi", action="store_true", help="Test OI data")
        
        args = parser.parse_args()
        
        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No valid stocks loaded[/red]")
            exit(1)
        
        if args.test_oi:
            console.print("🧪 [yellow]Testing OI data...[/yellow]")
            test_stocks = stocks[:10]
            stock_multi_data = prefetch_clean(test_stocks)
            console.print(f"✅ Test complete: [cyan]{len(oi_symbols_found)}[/cyan] symbols with OI")
            
        elif args.live:
            console.print("[yellow]🔴 Live mode coming soon...[/yellow]")
            
        elif args.backtest_date:
            try:
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                run_backtest_clean(args.backtest_date, stocks)
            except ValueError:
                console.print("[red]❌ Invalid date format. Use YYYY-MM-DD[/red]")
        else:
            console.print("\n🎯 [bold green]Option Signal Scanner v3.0[/bold green]")
            console.print("[cyan]Usage:[/cyan]")
            console.print("  [yellow]python scanner.py --backtest-date 2025-09-23[/yellow]")
            console.print("  [yellow]python scanner.py --live[/yellow]")
            console.print("  [yellow]python scanner.py --test-oi[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Scanner stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Error: {e}[/red]")
    finally:
        # Clean shutdown
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except Exception:
                pass
        
        if performance_metrics:
            total = sum(performance_metrics.values())
            if total > 0:
                console.print(f"\n📊 [bold green]Final: {total} signals generated[/bold green]")
        
        console.print("✅ [green]Shutdown complete[/green]")
# Enhanced OI Debugging Version - Part 2
from oi_debug_scanner_part1 import log_api_response_structure, enhanced_oi_column_detection, create_oi_debug_report, CONFIG, logger, oi_debug_info
# ... import other required elements ...

# ================= CORE OI INTEGRATION IN MAIN LOOP =================

def fetch_one_with_oi(symbol, timeframe_minutes):
    """Enhanced fetch function to log and debug API response"""
    # Use TrueData API
    td_session = TD_hist(username=CONFIG['TDUSERNAME'], password=CONFIG['TDPASSWORD'])
    raw_df = td_session.get_data(symbol=symbol, duration=CONFIG['DURATION_MAP'][timeframe_minutes], bar_size=CONFIG['BAR_SIZE_MAP'][timeframe_minutes])
    log_api_response_structure(symbol, raw_df, timeframe_minutes)
    return raw_df


def normalize_hist_df_with_oi(df, symbol):
    """Enhanced normalization with OI column checking"""
    oi_col, has_oi = enhanced_oi_column_detection(df, symbol)
    # If OI is present and meaningful, keep it
    if oi_col is not None and has_oi:
        df['normalized_oi'] = df[oi_col]
        logger.debug(f"Symbol {symbol}: Using OI column '{oi_col}' for analysis.")
    else:
        df['normalized_oi'] = 0.0
        logger.debug(f"Symbol {symbol}: No valid OI found. Using 0 as fallback.")
    return df


def analyze_signals_with_oi(df, symbol):
    """Show actual OI usage in main signal analysis"""
    # Debug: Print OI stats
    oi_mean = df['normalized_oi'].mean() if 'normalized_oi' in df.columns else 0.0
    logger.debug(f"Analyzing {symbol}: normalized_oi mean={oi_mean:.2f}")
    # Example signal logic...
    # [Replace with your main buy/sell analysis]
    momentum = df['normalized_oi'].iloc[-1] - df['normalized_oi'].iloc[-2] if len(df) >= 2 else 0
    if momentum > 0:
        logger.info(f"BUY signal for {symbol}: OI Momentum={momentum:.2f}")
    elif momentum < 0:
        logger.info(f"SELL signal for {symbol}: OI Momentum={momentum:.2f}")
    else:
        logger.info(f"NO ACTION for {symbol}: OI momentum flat.")


def main(symbols, timeframe_minutes):
    for symbol in symbols:
        df_raw = fetch_one_with_oi(symbol, timeframe_minutes)
        df_norm = normalize_hist_df_with_oi(df_raw, symbol)
        analyze_signals_with_oi(df_norm, symbol)
    # Print OI debugging report
    create_oi_debug_report()

# Entry point
if __name__ == "__main__":
    # Example: Run with a list of FNO stocks and 5 min interval
    symbols = ["RELIANCE", "TCS", "SBIN", "INFY", "HDFCBANK"]
    timeframe_minutes = 5
    main(symbols, timeframe_minutes)
