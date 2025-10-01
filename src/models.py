"""
Data models for Zion Stryker Trading Bot
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class CandleData(BaseModel):
    """Single candle/bar data"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class SignalData(BaseModel):
    """Trading signal data"""
    asset: str
    direction: str  # "buy" or "sell"
    timeframe: str
    urgency: Optional[str] = None  # "nice", "sharp", "hot" (Flash only) or None (Super)
    confidence: float
    reasons: List[str]
    mode: str  # "flash" or "super"
    timestamp: datetime
    expiry: Optional[int] = None

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    rsi: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    ema_6: Optional[float] = None
    ema_18: Optional[float] = None
    ema_proximity: Optional[float] = None
    adx: Optional[float] = None  # Added for future use

class SignalResponse(BaseModel):
    """API response for signals"""
    signals: List[SignalData]
    timestamp: str

class ScannerToggleRequest(BaseModel):
    """Scanner toggle request"""
    action: str  # "start" or "stop"
    mode: Optional[str] = "flash"
    timeframe: Optional[str] = "60"
