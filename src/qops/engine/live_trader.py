import json
import redis
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Initialize Alpaca and Redis
# (Paper trading = True)
trading_client = TradingClient("YOUR_API_KEY", "YOUR_SECRET_KEY", paper=True)
r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
pubsub = r.pubsub()

def listen_for_trades():
    print("🎧 Live Trader listening on Redis channel 'live_trade_signals'...")
    pubsub.subscribe("live_trade_signals")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            signal = json.loads(message['data'])
            print(f"\n🚨 RECEIVED SIGNAL: {signal['strategy']} (ML Score: {signal['ml_score']:.2f})")
            
            try:
                execute_alpaca_order(signal)
            except Exception as e:
                print(f"❌ Execution Failed: {e}")

def execute_alpaca_order(signal: dict):
    """Parses the Redis signal and builds the Alpaca Order."""
    print(f"🛠️ [STUB] Preparing to route order to Alpaca: {signal['action']}")
    print(f"Legs to route: {signal['legs']}")
    # Full multi-leg routing logic will go here once backtesting is verified.
    print("✅ Order routing simulation complete.")

if __name__ == "__main__":
    listen_for_trades()