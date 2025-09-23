"""
Data models for Zion Stryker Trading Bot
"""
from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime

class ScannerToggleRequest(BaseModel):
    """Request model for scanner toggle endpoint"""
    action: Literal["start", "stop"]
    mode: Optional[Literal["flash", "super"]] = "flash"
    timeframe: Optional[str] = "60"  # Default 1 minute

class SignalData(BaseModel):
    """Trading signal data model"""
    asset: str
    direction: Literal["buy", "sell"]
    priority: Literal["hot", "sharp", "nice"]
    mode: Literal["flash", "super"]
    timeframe: str
    timestamp: datetime
    confidence: float
    indicators: dict
    signal_id: str

class SignalResponse(BaseModel):
    """Response model for signals"""
    success: bool
    signal: Optional[SignalData] = None
    message: str
    timestamp: datetime

class CandleData(BaseModel):
    """Candle data model"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

class TechnicalIndicators(BaseModel):
    """Technical indicators for analysis"""
    rsi: Optional[float] = None
    ema_6: Optional[float] = None
    ema_18: Optional[float] = None
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