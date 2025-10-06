"""
Mock PocketOption API for development/testing with trade execution capability
"""
import asyncio
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from .models import TradeResult

class MockPocketOptionAsync:
    """Mock async PocketOption client with trade execution"""
    
    def __init__(self, ssid: str):
        self.ssid = ssid
        self.connected = False
        self.balance = 1000.0
        self.active_trades: Dict[str, TradeResult] = {}
    
    async def get_candles(self, asset: str, timeframe: int, count: int) -> List[Dict]:
        """Mock candle data generation"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Generate mock candle data with realistic patterns
        candles = []
        base_price = 1.1000 if "EUR" in asset else 150.00
        
        # Add some trend to make it more realistic
        trend_direction = random.choice([1, -1])
        trend_strength = random.uniform(0.0001, 0.0003)
        
        for i in range(count):
            timestamp = datetime.now() - timedelta(seconds=timeframe * (count - i))
            
            # Apply trend
            base_price += trend_direction * trend_strength
            
            # Generate realistic OHLC data
            open_price = base_price + random.uniform(-0.005, 0.005)
            close_price = open_price + random.uniform(-0.008, 0.008)
            high_price = max(open_price, close_price) + random.uniform(0, 0.004)
            low_price = min(open_price, close_price) - random.uniform(0, 0.004)
            
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
        """Get current balance"""
        await asyncio.sleep(0.05)
        return self.balance
    
    async def place_trade(
        self, 
        asset: str, 
        direction: str, 
        amount: float, 
        expiry_seconds: int
    ) -> TradeResult:
        """
        Place a binary options trade
        
        Args:
            asset: Trading asset (e.g., "EURUSD_otc")
            direction: "call" or "put"
            amount: Trade amount in dollars
            expiry_seconds: Expiration time in seconds
        
        Returns:
            TradeResult object with trade details
        """
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Generate trade ID
        trade_id = str(uuid.uuid4())[:8]
        
        # Get current price (simulate with random price)
        base_price = 1.1000 if "EUR" in asset else 150.00
        entry_price = base_price + random.uniform(-0.01, 0.01)
        
        # Create trade result
        trade = TradeResult(
            trade_id=trade_id,
            asset=asset,
            direction=direction,
            mode="flash",  # Will be updated by caller
            timeframe=str(expiry_seconds),
            amount=amount,
            entry_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(seconds=expiry_seconds),
            entry_price=entry_price,
            result="pending"
        )
        
        # Store active trade
        self.active_trades[trade_id] = trade
        
        # Schedule trade completion
        asyncio.create_task(self._complete_trade(trade_id, entry_price, expiry_seconds))
        
        return trade
    
    async def _complete_trade(self, trade_id: str, entry_price: float, expiry_seconds: int):
        """Complete a trade after expiry time"""
        await asyncio.sleep(expiry_seconds)
        
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        
        # Simulate exit price with 60% win rate (better than random)
        win_probability = 0.60
        is_win = random.random() < win_probability
        
        if trade.direction == "call":
            if is_win:
                exit_price = entry_price + random.uniform(0.0001, 0.001)
                trade.result = "win"
            else:
                exit_price = entry_price - random.uniform(0.0001, 0.001)
                trade.result = "loss"
        else:  # put
            if is_win:
                exit_price = entry_price - random.uniform(0.0001, 0.001)
                trade.result = "win"
            else:
                exit_price = entry_price + random.uniform(0.0001, 0.001)
                trade.result = "loss"
        
        trade.exit_price = exit_price
        trade.close_time = datetime.now()
        
        # Calculate profit/loss (85% payout for wins)
        if trade.result == "win":
            trade.profit = trade.amount * 0.85
            self.balance += trade.profit
        else:
            trade.profit = -trade.amount
            self.balance += trade.profit
    
    async def get_trade_result(self, trade_id: str) -> Optional[TradeResult]:
        """Get result of a specific trade"""
        await asyncio.sleep(0.05)
        return self.active_trades.get(trade_id)
    
    async def get_all_trades(self) -> List[TradeResult]:
        """Get all trades"""
        await asyncio.sleep(0.05)
        return list(self.active_trades.values())
    
    async def subscribe_symbol(self, asset: str):
        """Mock symbol subscription for real-time data"""
        await asyncio.sleep(0.1)
        
        async def mock_stream():
            base_price = 1.1000 if "EUR" in asset else 150.00
            while True:
                await asyncio.sleep(1)
                base_price += random.uniform(-0.0005, 0.0005)
                yield {
                    'open': base_price + random.uniform(-0.001, 0.001),
                    'high': base_price + random.uniform(0, 0.002),
                    'low': base_price - random.uniform(0, 0.002),
                    'close': base_price + random.uniform(-0.001, 0.001),
                    'volume': random.randint(100, 1000),
                    'timestamp': int(datetime.now().timestamp())
                }
        
        return mock_stream()
