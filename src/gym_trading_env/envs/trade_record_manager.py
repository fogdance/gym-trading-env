# src/gym_trading_env/envs/trade_record_manager.py

from decimal import Decimal
from datetime import datetime
from gym_trading_env.envs.trade_record import TradeRecord
import json

def custom_serializer(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, TradeRecord):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")

class TradeRecordManager:
    def __init__(self):
        self.trade_history = []

    def record_trade(self, trade_data):
        self.trade_history.append(trade_data)

    def dump_to_json(self, file_path):
        trade_history_json = json.dumps(self.trade_history, default=custom_serializer, indent=4)
        
        with open(file_path, 'w') as json_file:
            json_file.write(trade_history_json)
