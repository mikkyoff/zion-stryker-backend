"""
Trading Signal Engine for Zion Stryker
Implements Flash and Super mode trading strategies
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
try:
    import talib
except ImportError:
    # Mock talib functions for development
    class MockTalib:
        @staticmethod
        def RSI(data, timeperiod=14):
            return np.array([50.0 + np.random.uniform(-20, 20) for _ in range(len(data))])
        
        @staticmethod  
        def EMA(data, timeperiod=6):
            return np.array(data)  # Simplified mock
        
        @staticmethod
        def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2):
            mean = np.mean(data)
            std = np.std(data)
            upper = np.array([mean + std for _ in range(len(data))])
            middle = np.array([mean for _ in range(len(data))])
            lower = np.array([mean - std for _ in range(len(data))])
            return upper, middle, lower
        
        @staticmethod
        def ADX(high, low, close, timeperiod=14):
            return np.array([25.0 + np.random.uniform(-10, 10) for _ in range(len(close))])
        
        @staticmethod
        def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
            line = np.array([0.1 + np.random.uniform(-0.05, 0.05) for _ in range(len(data))])
            signal = np.array([0.08 + np.random.uniform(-0.05, 0.05) for _ in range(len(data))])
            histogram = line - signal
            return line, signal, histogram
        
        @staticmethod
        def ATR(high, low, close, timeperiod=14):
            return np.array([0.01 + np.random.uniform(-0.005, 0.005) for _ in range(len(close))])
    
    talib = MockTalib()
import logging
from datetime import datetime, timedelta
import math

from .models import CandleData, TechnicalIndicators, SignalData

logger = logging.getLogger(__name__)

class SignalEngine:
    """Core trading signal engine implementing Flash and Super mode strategies"""
    
    def __init__(self):
        logger.info("🔧 Initializing Signal Engine...")
        
    def calculate_indicators(self, candles: List[CandleData], timeframe: str) -> TechnicalIndicators:
        """Calculate technical indicators for given candle data"""
        if len(candles) < 50:  # Need sufficient data for indicators
            return TechnicalIndicators()
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([{
            'open': c.open,
            'high': c.high, 
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles])
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI settings - 14 for all timeframes
        rsi = talib.RSI(close, timeperiod=14)
        
        # EMA calculations
        ema_6 = talib.EMA(close, timeperiod=6)
        ema_18 = talib.EMA(close, timeperiod=18)
        
        # Bollinger Bands settings based on timeframe
        if timeframe == "30":
            bb_period, bb_dev = 14, 2
        else:
            bb_period, bb_dev = 20, 2
            
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev)
        
        # BB Width calculation
        bb_width = ((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]) * 100 if bb_middle[-1] != 0 else 0
        
        # ADX calculation
        adx = talib.ADX(high, low, close, timeperiod=14)
        
        # MACD calculation (12, 26, 9)
        macd_line, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # ATR calculation
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # STM (Smart Trend Momentum) - EMA 6 angle calculation
        stm_angle = self._calculate_ema_angle(ema_6)
        
        return TechnicalIndicators(
            rsi=float(rsi[-1]) if not np.isnan(rsi[-1]) else None,
            ema_6=float(ema_6[-1]) if not np.isnan(ema_6[-1]) else None,
            ema_18=float(ema_18[-1]) if not np.isnan(ema_18[-1]) else None,
            bb_upper=float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
            bb_middle=float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None,
            bb_lower=float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
            bb_width=bb_width,
            adx=float(adx[-1]) if not np.isnan(adx[-1]) else None,
            macd_line=float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None,
            macd_signal=float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
            macd_histogram=float(macd_histogram[-1]) if not np.isnan(macd_histogram[-1]) else None,
            atr=float(atr[-1]) if not np.isnan(atr[-1]) else None,
            stm_angle=stm_angle
        )
    
    def _calculate_ema_angle(self, ema_values: np.ndarray) -> Optional[float]:
        """Calculate EMA 6 angle for STM filter"""
        if len(ema_values) < 2 or np.isnan(ema_values[-1]) or np.isnan(ema_values[-2]):
            return None
        
        # Calculate angle in degrees
        price_change = ema_values[-1] - ema_values[-2]
        time_change = 1  # 1 period
        
        angle_radians = math.atan(price_change / (ema_values[-2] * 0.01))  # Normalize
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees
    
    def analyze_flash_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """Analyze Flash mode trading rules"""
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close
        
        # Flash mode rules validation
        bull_candles = self._count_bullish_candles(candles[-3:])  # Last 3 including current
        bear_candles = self._count_bearish_candles(candles[-3:])  # Last 3 including current
        
        # Check if market is choppy (filter out)
        if not self._is_trending_market(indicators):
            return None
        
        # Buy signal analysis
        if (bull_candles >= 3 and 
            indicators.bb_width and indicators.bb_width > 45 and  # BB expanding
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle):
            
            # Check 45% space to upper BB
            space_to_upper = ((indicators.bb_upper - current_price) / (indicators.bb_upper - indicators.bb_lower)) * 100
            
            if (space_to_upper >= 45 and 
                current_price > indicators.bb_lower and  # Above lower BB
                indicators.rsi and indicators.rsi > 50):  # RSI > 50
                
                # EMA cross analysis
                if (indicators.ema_6 and indicators.ema_18 and 
                    self._is_ema_cross_up(candles, indicators.ema_6, indicators.ema_18)):
                    
                    priority = self._determine_priority(indicators, "buy", "flash")
                    
                    return SignalData(
                        asset=asset,
                        direction="buy",
                        priority=priority,
                        mode="flash",
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        confidence=self._calculate_confidence(indicators, "buy"),
                        indicators=indicators.dict(),
                        signal_id=f"{asset}_{timeframe}_{int(datetime.now().timestamp())}"
                    )
        
        # Sell signal analysis
        if (bear_candles >= 3 and 
            indicators.bb_width and indicators.bb_width > 45 and  # BB expanding
            indicators.bb_upper and indicators.bb_lower and indicators.bb_middle):
            
            # Check 45% space to lower BB
            space_to_lower = ((current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)) * 100
            
            if (space_to_lower >= 45 and 
                current_price < indicators.bb_upper and  # Below upper BB
                indicators.rsi and indicators.rsi < 50):  # RSI < 50
                
                # EMA cross analysis
                if (indicators.ema_6 and indicators.ema_18 and 
                    self._is_ema_cross_down(candles, indicators.ema_6, indicators.ema_18)):
                    
                    priority = self._determine_priority(indicators, "sell", "flash")
                    
                    return SignalData(
                        asset=asset,
                        direction="sell",
                        priority=priority,
                        mode="flash",
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        confidence=self._calculate_confidence(indicators, "sell"),
                        indicators=indicators.dict(),
                        signal_id=f"{asset}_{timeframe}_{int(datetime.now().timestamp())}"
                    )
        
        return None
    
    def analyze_super_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """Analyze Super mode trading rules (includes all Flash rules plus additional filters)"""
        # First check Flash mode rules
        flash_signal = self.analyze_flash_mode(candles, timeframe, asset)
        if not flash_signal:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        
        # Additional Super mode filters
        
        # ADX Filter - stronger for longer timeframes
        adx_threshold = 20 if timeframe in ["30", "60"] else 25
        if not indicators.adx or indicators.adx < adx_threshold:
            return None
        
        # MACD Filter
        if not self._macd_filter_passed(indicators, flash_signal.direction):
            return None
        
        # Support/Resistance analysis
        if not self._support_resistance_check(candles, indicators, flash_signal.direction):
            return None
        
        # Fibonacci retracement (for 5m, 10m, 15m timeframes)
        if timeframe in ["300", "600", "900"]:  # 5m, 10m, 15m
            if not self._fibonacci_check(candles, flash_signal.direction):
                return None
        
        # Price action pattern confirmation
        if not self._price_action_confirmation(candles, flash_signal.direction):
            return None
        
        # Update to Super mode
        flash_signal.mode = "super"
        flash_signal.priority = self._determine_priority(indicators, flash_signal.direction, "super")
        
        return flash_signal
    
    def _count_bullish_candles(self, candles: List[CandleData]) -> int:
        """Count consecutive bullish candles"""
        return sum(1 for c in candles if c.close > c.open)
    
    def _count_bearish_candles(self, candles: List[CandleData]) -> int:
        """Count consecutive bearish candles"""
        return sum(1 for c in candles if c.close < c.open)
    
    def _is_trending_market(self, indicators: TechnicalIndicators) -> bool:
        """Check if market is trending (not choppy)"""
        if not indicators.bb_width:
            return False
        return indicators.bb_width > 20  # Minimum BB width for trending
    
    def _is_ema_cross_up(self, candles: List[CandleData], ema_6: float, ema_18: float) -> bool:
        """Check if EMA 6 is crossing above EMA 18"""
        if len(candles) < 2:
            return False
        
        # Current EMA 6 above EMA 18 and getting closer/crossing
        return ema_6 > ema_18 and abs(ema_6 - ema_18) / ema_18 < 0.02  # Within 2%
    
    def _is_ema_cross_down(self, candles: List[CandleData], ema_6: float, ema_18: float) -> bool:
        """Check if EMA 6 is crossing below EMA 18"""
        if len(candles) < 2:
            return False
        
        # Current EMA 6 below EMA 18 and getting closer/crossing
        return ema_6 < ema_18 and abs(ema_6 - ema_18) / ema_18 < 0.02  # Within 2%
    
    def _determine_priority(self, indicators: TechnicalIndicators, direction: str, mode: str) -> str:
        """Determine signal priority: hot, sharp, nice"""
        
        # Hot: All rules align + STM filter
        if self._stm_filter_check(indicators, direction):
            return "hot"
        
        # Sharp: Core rules (ADX + EMA curve + STM)
        if mode == "super" and indicators.adx and indicators.adx > 25:
            return "sharp"
        
        # Nice: Flash-only rules
        return "nice"
    
    def _stm_filter_check(self, indicators: TechnicalIndicators, direction: str) -> bool:
        """Smart Trend Momentum filter check"""
        if not indicators.stm_angle:
            return False
        
        if direction == "buy":
            return indicators.stm_angle >= 15.0
        else:  # sell
            return indicators.stm_angle <= -15.0
    
    def _macd_filter_passed(self, indicators: TechnicalIndicators, direction: str) -> bool:
        """MACD filter for Super mode"""
        if not all([indicators.macd_line, indicators.macd_histogram]):
            return False
        
        if direction == "buy":
            return indicators.macd_line > 0 and indicators.macd_histogram > 0
        else:  # sell
            return indicators.macd_line < 0 and indicators.macd_histogram < 0
    
    def _support_resistance_check(self, candles: List[CandleData], indicators: TechnicalIndicators, direction: str) -> bool:
        """Support/Resistance zone analysis"""
        if len(candles) < 20:
            return False
        
        # Simplified S/R check using recent highs/lows
        recent_candles = candles[-20:]
        current_price = candles[-1].close
        
        if direction == "buy":
            # Price should be near support level
            recent_lows = [c.low for c in recent_candles]
            support_level = min(recent_lows)
            return abs(current_price - support_level) / current_price < 0.02
        else:  # sell
            # Price should be near resistance level  
            recent_highs = [c.high for c in recent_candles]
            resistance_level = max(recent_highs)
            return abs(current_price - resistance_level) / current_price < 0.02
    
    def _fibonacci_check(self, candles: List[CandleData], direction: str) -> bool:
        """Fibonacci retracement analysis for 5m+ timeframes"""
        if len(candles) < 50:
            return False
        
        # Find recent swing high/low
        recent_prices = [c.close for c in candles[-50:]]
        current_price = candles[-1].close
        
        swing_high = max(recent_prices)
        swing_low = min(recent_prices)
        
        # Calculate Fib levels
        fib_382 = swing_low + 0.382 * (swing_high - swing_low)
        fib_500 = swing_low + 0.500 * (swing_high - swing_low)
        fib_618 = swing_low + 0.618 * (swing_high - swing_low)
        
        # Check if price is near key Fib level (within 1%)
        tolerance = 0.01
        for fib_level in [fib_382, fib_500, fib_618]:
            if abs(current_price - fib_level) / current_price < tolerance:
                return True
        
        return False
    
    def _price_action_confirmation(self, candles: List[CandleData], direction: str) -> bool:
        """Price action pattern confirmation"""
        if len(candles) < 2:
            return False
        
        current_candle = candles[-1]
        prev_candle = candles[-2]
        
        # Calculate average body size
        avg_body_size = np.mean([abs(c.close - c.open) for c in candles[-5:]])
        current_body_size = abs(current_candle.close - current_candle.open)
        
        # Body must be at least average size
        if current_body_size < avg_body_size:
            return False
        
        # Check for engulfing patterns
        if direction == "buy":
            # Bullish engulfing
            return (prev_candle.close < prev_candle.open and  # Previous bearish
                   current_candle.close > current_candle.open and  # Current bullish
                   current_candle.close > prev_candle.open and
                   current_candle.open < prev_candle.close)
        else:  # sell
            # Bearish engulfing
            return (prev_candle.close > prev_candle.open and  # Previous bullish
                   current_candle.close < current_candle.open and  # Current bearish
                   current_candle.close < prev_candle.open and
                   current_candle.open > prev_candle.close)
    
    def _calculate_confidence(self, indicators: TechnicalIndicators, direction: str) -> float:
        """Calculate signal confidence score (0-1)"""
        confidence = 0.5  # Base confidence
        
        # Add confidence based on indicator strength
        if indicators.rsi:
            if direction == "buy" and indicators.rsi > 50:
                confidence += 0.1
            elif direction == "sell" and indicators.rsi < 50:
                confidence += 0.1
        
        if indicators.bb_width and indicators.bb_width > 50:
            confidence += 0.1
        
        if indicators.adx and indicators.adx > 30:
            confidence += 0.1
        
        if self._stm_filter_check(indicators, direction):
            confidence += 0.2
        
        return min(confidence, 1.0)