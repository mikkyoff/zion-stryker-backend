"""
Data models for Zion Stryker Trading Bot
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum

class CandleData(BaseModel):
    """Individual candle data point"""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: datetime

class TechnicalIndicators(BaseModel):
    """Technical indicators calculated from candle data"""
    rsi: Optional[float] = None
    ema_6: Optional[float] = None
    ema_18: Optional[float] = None
    sma_7: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    adx: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    stm_angle: Optional[float] = None

class SignalData(BaseModel):
    """Trading signal information"""
    asset: str
    direction: Literal["call", "put"]
    mode: Literal["flash", "super"]
    timeframe: str
    confidence: float
    priority: Literal["high", "medium", "low"]
    timestamp: datetime
    indicators: Optional[TechnicalIndicators] = None
    reason: Optional[str] = None

class TradeResult(BaseModel):
    """Trade execution result"""
    trade_id: str
    asset: str
    direction: Literal["call", "put"]
    mode: Literal["flash", "super"]
    timeframe: str
    amount: float
    entry_time: datetime
    expiry_time: datetime
    entry_price: float
    exit_price: Optional[float] = None
    result: Optional[Literal["win", "loss", "pending"]] = "pending"
    profit: Optional[float] = None
    close_time: Optional[datetime] = None

class ScannerToggleRequest(BaseModel):
    """Scanner toggle request"""
    enabled: bool
    mode: Literal["flash", "super"] = "flash"
    timeframe: str = "30"

class SignalResponse(BaseModel):
    """API response for signals"""
    signals: List[SignalData]
    count: int
    timestamp: datetime

class TradeResultsResponse(BaseModel):
    """API response for trade results"""
    trades: List[TradeResult]
    total_trades: int
    wins: int
    losses: int
    pending: int
    win_rate: float
    total_profit: float
