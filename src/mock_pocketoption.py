"""
Mock PocketOption API for development/testing when BinaryOptionsToolsV2 is not available
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict

class MockPocketOptionAsync:
    """Mock async PocketOption client for development"""
    
    def __init__(self, ssid: str):
        self.ssid = ssid
        self.connected = False
    
    async def get_candles(self, asset: str, timeframe: int, count: int) -> List[Dict]:
        """Mock candle data generation"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Generate mock candle data
        candles = []
        base_price = 1.1000 if "EUR" in asset else 150.00
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(seconds=timeframe * (count - i))
            
            # Generate realistic OHLC data
            open_price = base_price + random.uniform(-0.01, 0.01)
            close_price = open_price + random.uniform(-0.005, 0.005)
            high_price = max(open_price, close_price) + random.uniform(0, 0.003)
            low_price = min(open_price, close_price) - random.uniform(0, 0.003)
            
            candles.append({
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': random.randint(100, 1000),
                'timestamp': int(timestamp.timestamp())
            })
            
            base_price = close_price  # Use close as next open
        
        return candles
    
    async def get_balance(self) -> float:
        """Mock balance"""
        await asyncio.sleep(0.1)
        return 1000.0
    
    async def subscribe_symbol(self, asset: str):
        """Mock symbol subscription"""
        await asyncio.sleep(0.1)
        # Return async generator for streaming data
        async def mock_stream():
            while True:
                await asyncio.sleep(1)
                yield {
                    'open': 1.1000 + random.uniform(-0.01, 0.01),
                    'high': 1.1050 + random.uniform(-0.01, 0.01),
                    'low': 1.0950 + random.uniform(-0.01, 0.01),
                    'close': 1.1000 + random.uniform(-0.01, 0.01),
                    'volume': random.randint(100, 1000),
                    'timestamp': int(datetime.now().timestamp())
                }
        
        return mock_stream()