"""
Signal Engine for Zion Stryker Trading Bot
Implements Flash and Super mode trading strategies
"""
import numpy as np
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime

from .models import CandleData, SignalData, TechnicalIndicators

logger = logging.getLogger(__name__)

class SignalEngine:
    """Advanced signal generation engine with multiple strategies"""
    
    def __init__(self):
        self.min_candles = 50
    
    def calculate_indicators(self, candles: List[CandleData], timeframe: str) -> TechnicalIndicators:
        """Calculate technical indicators"""
        try:
            # Convert to pandas
            df = pd.DataFrame([{
                'close': c.close,
                'high': c.high,
                'low': c.low,
                'open': c.open
            } for c in candles])
            
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (20 period, 2 std)
            bb_period = 20
            bb_std = 2
            sma = df['close'].rolling(window=bb_period).mean()
            std = df['close'].rolling(window=bb_period).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)
            bb_width = ((bb_upper - bb_lower) / sma * 100)
            
            # EMAs
            ema_6 = df['close'].ewm(span=6, adjust=False).mean()
            ema_18 = df['close'].ewm(span=18, adjust=False).mean()
            
            # EMA proximity (percentage difference)
            ema_proximity = abs(ema_6.iloc[-1] - ema_18.iloc[-1]) / ema_18.iloc[-1] * 100
            
            return TechnicalIndicators(
                rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                bb_upper=float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                bb_middle=float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                bb_lower=float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                bb_width=float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else None,
                ema_6=float(ema_6.iloc[-1]) if not pd.isna(ema_6.iloc[-1]) else None,
                ema_18=float(ema_18.iloc[-1]) if not pd.isna(ema_18.iloc[-1]) else None,
                ema_proximity=float(ema_proximity) if not pd.isna(ema_proximity) else None
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return TechnicalIndicators()
    
    def _count_bullish_candles(self, candles: List[CandleData]) -> int:
        """Count consecutive bullish candles"""
        count = 0
        for candle in reversed(candles):
            if candle.close > candle.open:
                count += 1
            else:
                break
        return count
    
    def _count_bearish_candles(self, candles: List[CandleData]) -> int:
        """Count consecutive bearish candles"""
        count = 0
        for candle in reversed(candles):
            if candle.close < candle.open:
                count += 1
            else:
                break
        return count
    
    def analyze_flash_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Flash Mode Strategy (Fast Signals):
        - 3 consecutive candles in same direction
        - Price in lower/upper 50% of BB range
        - RSI confirmation
        - EMA cross confirmation
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close
        
        bull_candles = self._count_bullish_candles(candles[-3:])
        bear_candles = self._count_bearish_candles(candles[-3:])
        
        # BUY SIGNAL CONDITIONS
        if (bull_candles >= 3 and
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle and
            indicators.rsi and indicators.rsi > 50 and
            indicators.ema_6 and indicators.ema_18 and
            indicators.ema_6 > indicators.ema_18 and
            indicators.ema_proximity and indicators.ema_proximity <= 2.0):
            
            # Calculate position within BB bands
            bb_range = indicators.bb_upper - indicators.bb_lower
            position_in_bb = ((current_price - indicators.bb_lower) / bb_range) * 100
            
            # Distance to upper BB (room to grow)
            distance_to_upper = ((indicators.bb_upper - current_price) / current_price) * 100
            
            # Must be in lower 50% of BB range
            if position_in_bb < 50:
                reasons = [
                    f"{bull_candles} bullish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"BB Position: {position_in_bb:.0f}%",
                    f"Room to upper BB: {distance_to_upper:.1f}%",
                    f"EMA cross confirmed"
                ]
                
                # Hot signal if in lower 30% with >2% room
                is_hot = position_in_bb < 30 and distance_to_upper > 2.0
                
                return SignalData(
                    asset=asset,
                    direction="buy",
                    timeframe=timeframe,
                    urgency="hot" if is_hot else "normal",
                    confidence=min(95, 60 + (distance_to_upper * 8)),
                    reasons=reasons,
                    mode="flash",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        # SELL SIGNAL CONDITIONS
        if (bear_candles >= 3 and
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle and
            indicators.rsi and indicators.rsi < 50 and
            indicators.ema_6 and indicators.ema_18 and
            indicators.ema_6 < indicators.ema_18 and
            indicators.ema_proximity and indicators.ema_proximity <= 2.0):
            
            # Calculate position within BB bands
            bb_range = indicators.bb_upper - indicators.bb_lower
            position_in_bb = ((current_price - indicators.bb_lower) / bb_range) * 100
            
            # Distance to lower BB (room to fall)
            distance_to_lower = ((current_price - indicators.bb_lower) / current_price) * 100
            
            # Must be in upper 50% of BB range (>50%)
            if position_in_bb > 50:
                reasons = [
                    f"{bear_candles} bearish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"BB Position: {position_in_bb:.0f}%",
                    f"Room to lower BB: {distance_to_lower:.1f}%",
                    f"EMA cross confirmed"
                ]
                
                # Hot signal if in upper 70% with >2% room
                is_hot = position_in_bb > 70 and distance_to_lower > 2.0
                
                return SignalData(
                    asset=asset,
                    direction="sell",
                    timeframe=timeframe,
                    urgency="hot" if is_hot else "normal",
                    confidence=min(95, 60 + (distance_to_lower * 8)),
                    reasons=reasons,
                    mode="flash",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        return None
    
    def analyze_super_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Super Mode Strategy (High Confidence):
        - 3 consecutive candles (same as Flash)
        - Price in lower/upper 50% of BB range
        - Stricter RSI and EMA requirements
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close
        
        bull_candles = self._count_bullish_candles(candles[-3:])
        bear_candles = self._count_bearish_candles(candles[-3:])
        
        # BUY SIGNAL CONDITIONS (Super Mode)
        if (bull_candles >= 3 and
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle and
            indicators.rsi and indicators.rsi > 55 and
            indicators.ema_6 and indicators.ema_18 and
            indicators.ema_6 > indicators.ema_18 and
            indicators.ema_proximity and indicators.ema_proximity <= 1.0):
            
            bb_range = indicators.bb_upper - indicators.bb_lower
            position_in_bb = ((current_price - indicators.bb_lower) / bb_range) * 100
            distance_to_upper = ((indicators.bb_upper - current_price) / current_price) * 100
            
            # Must be in lower 50% of BB range
            if position_in_bb < 50:
                reasons = [
                    f"{bull_candles} bullish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"BB Position: {position_in_bb:.0f}%",
                    f"Room to upper BB: {distance_to_upper:.1f}%",
                    f"Tight EMA cross: {indicators.ema_proximity:.1f}%"
                ]
                
                is_hot = position_in_bb < 30 and distance_to_upper > 2.5
                
                return SignalData(
                    asset=asset,
                    direction="buy",
                    timeframe=timeframe,
                    urgency="hot" if is_hot else "normal",
                    confidence=min(98, 70 + (distance_to_upper * 7)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        # SELL SIGNAL CONDITIONS (Super Mode)
        if (bear_candles >= 3 and
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle and
            indicators.rsi and indicators.rsi < 45 and
            indicators.ema_6 and indicators.ema_18 and
            indicators.ema_6 < indicators.ema_18 and
            indicators.ema_proximity and indicators.ema_proximity <= 1.0):
            
            bb_range = indicators.bb_upper - indicators.bb_lower
            position_in_bb = ((current_price - indicators.bb_lower) / bb_range) * 100
            distance_to_lower = ((current_price - indicators.bb_lower) / current_price) * 100
            
            # Must be in upper 50% of BB range (>50%)
            if position_in_bb > 50:
                reasons = [
                    f"{bear_candles} bearish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"BB Position: {position_in_bb:.0f}%",
                    f"Room to lower BB: {distance_to_lower:.1f}%",
                    f"Tight EMA cross: {indicators.ema_proximity:.1f}%"
                ]
                
                is_hot = position_in_bb > 70 and distance_to_lower > 2.5
                
                return SignalData(
                    asset=asset,
                    direction="sell",
                    timeframe=timeframe,
                    urgency="hot" if is_hot else "normal",
                    confidence=min(98, 70 + (distance_to_lower * 7)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        return None