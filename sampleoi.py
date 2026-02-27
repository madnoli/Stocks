import requests
import datetime

# Configuration - replace with your own TrueData credentials
USERNAME = "tdwsp751"
PASSWORD = "raj@751"
BASE_URL = "https://api.truedata.in"  # Adjust base URL from documentation

def get_access_token(username, password):
    url = f"{BASE_URL}/oauth2/token"
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password,
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    return resp.json()['access_token']

def get_historical_data(token, symbol, start_dt, end_dt, interval="5min"):
    url = f"{BASE_URL}/history/data"
    headers = {'Authorization': f'Bearer {token}'}
    params = {
        'symbol': symbol,
        'start': start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        'end': end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        'interval': interval,
        'format': 'json',
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def get_option_chain(token, symbol):
    url = f"{BASE_URL}/options/chain"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'symbol': symbol, 'format': 'json'}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()

def compute_pcr(option_chain):
    total_pe_oi = sum([strike['pe']['oi'] for strike in option_chain if 'pe' in strike])
    total_ce_oi = sum([strike['ce']['oi'] for strike in option_chain if 'ce' in strike])
    if total_ce_oi == 0:
        return None
    return total_pe_oi / total_ce_oi

def main():
    token = get_access_token(USERNAME, PASSWORD)
    now = datetime.datetime.now()
    start = now - datetime.timedelta(days=1)

    for symbol in ["HDFCBANK", "RELIANCE"]:
        print(f"Fetching historical data for {symbol}...")
        bars = get_historical_data(token, symbol, start, now)
        if bars and 'data' in bars:
            last_bar = bars['data'][-1]
            print(f"Open: {last_bar['open']}")
            print(f"High: {last_bar['high']}")
            print(f"Low: {last_bar['low']}")
            print(f"Close: {last_bar['close']}")
            print(f"Volume: {last_bar['volume']}")
            print(f"Open Interest: {last_bar.get('oi', 'n/a')}")
        else:
            print("No historical data found.")

        print(f"Fetching option chain for {symbol}...")
        chain = get_option_chain(token, symbol)
        if chain and 'strikes' in chain:
            pcr = compute_pcr(chain['strikes'])
            print(f"Put-Call Ratio (PCR): {pcr if pcr else 'n/a'}")
        else:
            print("No option chain data found.")

        print("-" * 40)

if __name__ == "__main__":
    main()
