"""
Roostoo Mock Exchange Public API - Python Client
------------------------------------------------
This script demonstrates how to interact with the Roostoo Mock Exchange API.
It supports both public and signed endpoints with HMAC SHA256 authentication.

Base URL: https://mock-api.roostoo.com
"""

import requests
import time
import hmac
import hashlib

# --- API Configuration ---
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = "yKnzOuJFVge4Rh2VwXRaw5JVGw8Vd1f8msaln4WkuhljCDSgcXw6DSaHXso5iguk"      # Replace with your actual API key
SECRET_KEY = "Dw0iFkGHUR6cul3Wgr7C7Q6r4SxELO0zthTNUH9S5VCGlrBooNOVDdm69wXp2AV6"  # Replace with your actual secret key

  

# ------------------------------
# Utility Functions
# ------------------------------

def _get_timestamp():
    """Return a 13-digit millisecond timestamp as string."""
    return str(int(time.time() * 1000))


def _get_signed_headers(payload: dict = {}):
    """
    Generate signed headers and totalParams for RCL_TopLevelCheck endpoints.
    """
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }

    return headers, payload, total_params


# ------------------------------
# Public Endpoints
# ------------------------------

def check_server_time():
    """Check API server time."""
    url = f"{BASE_URL}/v3/serverTime"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking server time: {e}")
        return None


def get_exchange_info():
    """Get exchange trading pairs and info."""
    url = f"{BASE_URL}/v3/exchangeInfo"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting exchange info: {e}")
        return None


def get_ticker(pair=None):
    """Get ticker for one or all pairs."""
    url = f"{BASE_URL}/v3/ticker"
    params = {'timestamp': _get_timestamp()}
    if pair:
        params['pair'] = pair
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting ticker: {e}")
        return None


# ------------------------------
# Signed Endpoints
# ------------------------------

def get_balance():
    """Get wallet balances (RCL_TopLevelCheck)."""
    url = f"{BASE_URL}/v3/balance"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting balance: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def get_pending_count():
    """Get total pending order count."""
    url = f"{BASE_URL}/v3/pending_count"
    headers, payload, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params=payload)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting pending count: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    """
    Place a LIMIT or MARKET order.
    """
    url = f"{BASE_URL}/v3/place_order"
    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"

    if order_type == 'LIMIT' and price is None:
        print("Error: LIMIT orders require 'price'.")
        return None

    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(quantity)
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error placing order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def query_order(order_id=None, pair=None, pending_only=None):
    """Query order history or pending orders."""
    url = f"{BASE_URL}/v3/query_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


def cancel_order(order_id=None, pair=None):
    """Cancel specific or all pending orders."""
    url = f"{BASE_URL}/v3/cancel_order"
    payload = {}
    if order_id:
        payload['order_id'] = str(order_id)
    elif pair:
        payload['pair'] = pair

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error canceling order: {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
        return None


# ------------------------------
# Quick Demo Section
# ------------------------------
if __name__ == "__main__":
    print("\n--- Checking Server Time ---")
    print(check_server_time())

    print("\n--- Getting Exchange Info ---")
    info = get_exchange_info()
    if info:
        print(f"Available Pairs: {list(info.get('TradePairs', {}).keys())}")

    print("\n--- Getting Market Ticker (BTC/USD) ---")
    ticker = get_ticker("BTC/USD")
    if ticker:
        print(ticker.get("Data", {}).get("BTC/USD", {}))

    print("\n--- Getting Account Balance ---")
    print(get_balance())

    print("\n--- Checking Pending Orders ---")
    print(get_pending_count())

    # Uncomment these to test trading actions:
    # print(place_order("BTC", "BUY", 0.01, price=95000))  # LIMIT
    print(place_order("BNB/USD", "BUY", 1))      
    print(place_order("BNB/USD", "SELL", 1))             # MARKET       
    print(query_order(pair="BNB/USD", pending_only=False))
    # print(cancel_order(pair="BNB/USD"))
