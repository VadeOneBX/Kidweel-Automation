import json
import redis
from datetime import datetime

# Initialize Redis (points to your existing Docker container)
r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def publish_trade_signal(top_trade: dict, strategy_mode: str):
    """Broadcasts the ML-approved trade to the execution node."""
    
    signal = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy": strategy_mode,
        "action": "EXECUTE_SPREAD" if "SPREAD" in strategy_mode else "EXECUTE_LONG",
        "legs": [],
        "ml_score": top_trade['ml_win_prob'],
        "expected_premium": top_trade.get('premium', 0.0)
    }
    
    # Format the legs for Alpaca based on Option Alpha logic
    if "SPREAD" in strategy_mode:
        signal["legs"] = [
            {"symbol": top_trade['short_symbol'], "side": "sell", "ratio": 1},
            {"symbol": top_trade['long_symbol'], "side": "buy", "ratio": 1}
        ]
    else:
        signal["legs"] = [
            {"symbol": top_trade['symbol'], "side": "buy", "ratio": 1}
        ]

    # Publish to the Redis channel
    r.publish("live_trade_signals", json.dumps(signal))
    print(f"📡 Broadcasted {strategy_mode} signal to Redis execution queue.")