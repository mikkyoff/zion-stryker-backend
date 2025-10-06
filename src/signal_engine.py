"""
Trading Signal Engine for Zion Stryker
Implements Flash and Super mode trading strategies for EURUSD_otc
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
            # Simple EMA calculation
            if len(data) < timeperiod:
                return np.array(data)
            alpha = 2 / (timeperiod + 1)
            ema = np.zeros(len(data))
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema
        
        @staticmethod
        def SMA(data, timeperiod=7):
            # Simple moving average
            result = np.zeros(len(data))
            for i in range(len(data)):
                if i < timeperiod - 1:
                    result[i] = np.mean(data[:i+1])
                else:
                    result[i] = np.mean(data[i-timeperiod+1:i+1])
            return result
        
        @staticmethod
        def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2):
            mean = np.mean(data[-timeperiod:]) if len(data) >= timeperiod else np.mean(data)
            std = np.std(data[-timeperiod:]) if len(data) >= timeperiod else np.std(data)
            upper = np.array([mean + nbdevup * std for _ in range(len(data))])
            middle = np.array([mean for _ in range(len(data))])
            lower = np.array([mean - nbdevdn * std for _ in range(len(data))])
            return upper, middle, lower
        
        @staticmethod
        def ADX(high, low, close, timeperiod=14):
            return np.array([25.0 + np.random.uniform(-10, 10) for _ in range(len(close))])
        
        @staticmethod
        def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
            # Simple MACD calculation
            ema_fast = MockTalib.EMA(data, fastperiod)
            ema_slow = MockTalib.EMA(data, slowperiod)
            macd_line = ema_fast - ema_slow
            signal = MockTalib.EMA(macd_line, signalperiod)
            histogram = macd_line - signal
            return macd_line, signal, histogram
        
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
        
        # RSI settings - timeframe specific
        if timeframe == "30":  # 30s timeframe
            rsi = talib.RSI(close, timeperiod=14)
        elif timeframe == "15":  # 15s timeframe
            rsi = talib.RSI(close, timeperiod=10)
        else:
            rsi = talib.RSI(close, timeperiod=14)
        
        # EMA calculations
        ema_6 = talib.EMA(close, timeperiod=6)
        ema_18 = talib.EMA(close, timeperiod=18)
        
        # SMA 7 for Super mode
        sma_7 = talib.SMA(close, timeperiod=7)
        
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
            sma_7=float(sma_7[-1]) if not np.isnan(sma_7[-1]) else None,
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
        if len(ema_values) < 2:
            return None
        
        # Calculate angle of last 2 EMA points
        y_diff = ema_values[-1] - ema_values[-2]
        angle = math.degrees(math.atan(y_diff))
        return round(angle, 2)
    
    def _is_doji(self, candle: CandleData) -> bool:
        """Check if candle is a doji (body <= 30% of range)"""
        candle_range = candle.high - candle.low
        if candle_range == 0:
            return True
        
        body = abs(candle.close - candle.open)
        body_percentage = (body / candle_range) * 100
        
        return body_percentage <= 30
    
    def _check_consecutive_candles(self, candles: List[CandleData], direction: str, count: int = 3) -> bool:
        """Check if last N candles are all in same direction (no doji)"""
        if len(candles) < count:
            return False
        
        last_candles = candles[-count:]
        
        for candle in last_candles:
            # Check if it's a doji
            if self._is_doji(candle):
                return False
            
            # Check direction
            if direction == "call":
                if candle.close <= candle.open:  # Not bullish
                    return False
            else:  # put
                if candle.close >= candle.open:  # Not bearish
                    return False
        
        return True
    
    def _calculate_bb_distance(self, price: float, bb_upper: float, bb_lower: float, direction: str) -> float:
        """Calculate distance from BB band as percentage"""
        bb_range = bb_upper - bb_lower
        if bb_range == 0:
            return 0
        
        if direction == "call":
            distance = ((bb_lower - price) / bb_range) * 100
        else:  # put
            distance = ((price - bb_upper) / bb_range) * 100
        
        return abs(distance)
    
    def analyze_flash_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Flash Mode Strategy:
        - 3 consecutive candles in same direction (no doji)
        - RSI validation (14 on 30s, 10 on 15s)
        - 35% BB distance
        - EMA 6/18 crossover as trigger
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        last_candle = candles[-1]
        
        # Determine direction based on EMA crossover
        if indicators.ema_6 is None or indicators.ema_18 is None:
            return None
        
        # Check for crossover
        prev_ema_6 = talib.EMA(np.array([c.close for c in candles]), timeperiod=6)[-2]
        prev_ema_18 = talib.EMA(np.array([c.close for c in candles]), timeperiod=18)[-2]
        
        direction = None
        if prev_ema_6 <= prev_ema_18 and indicators.ema_6 > indicators.ema_18:
            direction = "call"  # Bullish crossover
        elif prev_ema_6 >= prev_ema_18 and indicators.ema_6 < indicators.ema_18:
            direction = "put"  # Bearish crossover
        
        if not direction:
            return None
        
        # Rule 1: Check 3 consecutive candles
        if not self._check_consecutive_candles(candles, direction, count=3):
            return None
        
        # Rule 2: RSI validation
        if indicators.rsi is None:
            return None
        
        if direction == "call":
            if indicators.rsi >= 70:  # Overbought
                return None
        else:  # put
            if indicators.rsi <= 30:  # Oversold
                return None
        
        # Rule 3: No doji in last 3 candles (already checked)
        
        # Rule 4: BB distance check (35%)
        if indicators.bb_upper is None or indicators.bb_lower is None:
            return None
        
        bb_distance = self._calculate_bb_distance(
            last_candle.close, 
            indicators.bb_upper, 
            indicators.bb_lower, 
            direction
        )
        
        if bb_distance < 35:
            return None
        
        # Calculate confidence based on indicators
        confidence = self._calculate_confidence(indicators, direction, "flash")
        priority = self._determine_priority(confidence)
        
        return SignalData(
            asset=asset,
            direction=direction,
            mode="flash",
            timeframe=timeframe,
            confidence=confidence,
            priority=priority,
            timestamp=datetime.now(),
            indicators=indicators,
            reason=f"EMA crossover + {bb_distance:.1f}% BB distance + RSI {indicators.rsi:.1f}"
        )
    
    def analyze_super_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Super Mode Strategy:
        - All Flash mode rules apply
        - 3 consecutive candles, RSI check, no doji, 35% BB distance
        - SMA 7 validation (price must be on correct side)
        - MACD (12,26,9) crossover as trigger
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        last_candle = candles[-1]
        
        # Determine direction based on MACD crossover
        if indicators.macd_line is None or indicators.macd_signal is None:
            return None
        
        # Check for MACD crossover
        close_prices = np.array([c.close for c in candles])
        prev_macd_line, prev_macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        
        direction = None
        if prev_macd_line[-2] <= prev_macd_signal[-2] and indicators.macd_line > indicators.macd_signal:
            direction = "call"  # Bullish crossover
        elif prev_macd_line[-2] >= prev_macd_signal[-2] and indicators.macd_line < indicators.macd_signal:
            direction = "put"  # Bearish crossover
        
        if not direction:
            return None
        
        # Rule 1: Check 3 consecutive candles (from Flash)
        if not self._check_consecutive_candles(candles, direction, count=3):
            return None
        
        # Rule 2: RSI validation (from Flash)
        if indicators.rsi is None:
            return None
        
        if direction == "call":
            if indicators.rsi >= 70:  # Overbought
                return None
        else:  # put
            if indicators.rsi <= 30:  # Oversold
                return None
        
        # Rule 3: SMA 7 validation
        if indicators.sma_7 is None:
            return None
        
        if direction == "call":
            if last_candle.close < indicators.sma_7:  # Price must be above SMA
                return None
        else:  # put
            if last_candle.close > indicators.sma_7:  # Price must be below SMA
                return None
        
        # Rule 4: BB distance check (35%)
        if indicators.bb_upper is None or indicators.bb_lower is None:
            return None
        
        bb_distance = self._calculate_bb_distance(
            last_candle.close, 
            indicators.bb_upper, 
            indicators.bb_lower, 
            direction
        )
        
        if bb_distance < 35:
            return None
        
        # Calculate confidence based on indicators
        confidence = self._calculate_confidence(indicators, direction, "super")
        priority = self._determine_priority(confidence)
        
        return SignalData(
            asset=asset,
            direction=direction,
            mode="super",
            timeframe=timeframe,
            confidence=confidence,
            priority=priority,
            timestamp=datetime.now(),
            indicators=indicators,
            reason=f"MACD crossover + SMA7 + {bb_distance:.1f}% BB distance + RSI {indicators.rsi:.1f}"
        )
    
    def _calculate_confidence(self, indicators: TechnicalIndicators, direction: str, mode: str) -> float:
        """Calculate signal confidence score (0-100)"""
        confidence = 50.0  # Base confidence
        
        # RSI contribution
        if indicators.rsi:
            if direction == "call":
                if 40 <= indicators.rsi <= 60:
                    confidence += 15
                elif indicators.rsi < 40:
                    confidence += 10
            else:  # put
                if 40 <= indicators.rsi <= 60:
                    confidence += 15
                elif indicators.rsi > 60:
                    confidence += 10
        
        # ADX contribution (trend strength)
        if indicators.adx:
            if indicators.adx > 25:
                confidence += min((indicators.adx - 25) / 2, 15)
        
        # STM angle contribution
        if indicators.stm_angle:
            if direction == "call" and indicators.stm_angle > 0:
                confidence += min(indicators.stm_angle / 2, 10)
            elif direction == "put" and indicators.stm_angle < 0:
                confidence += min(abs(indicators.stm_angle) / 2, 10)
        
        # Super mode gets bonus confidence
        if mode == "super":
            confidence += 10
        
        return min(confidence, 100.0)
    
    def _determine_priority(self, confidence: float) -> str:
        """Determine signal priority based on confidence"""
        if confidence >= 75:
            return "high"
        elif confidence >= 60:
            return "medium"
        else:
            return "low"
