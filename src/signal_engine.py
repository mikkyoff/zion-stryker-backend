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
            
            # ADX calculation (14 period)
            adx_val = self._calculate_adx(df, 14)
            
            return TechnicalIndicators(
                rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                bb_upper=float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else None,
                bb_middle=float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None,
                bb_lower=float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else None,
                bb_width=float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else None,
                ema_6=float(ema_6.iloc[-1]) if not pd.isna(ema_6.iloc[-1]) else None,
                ema_18=float(ema_18.iloc[-1]) if not pd.isna(ema_18.iloc[-1]) else None,
                ema_proximity=float(ema_proximity) if not pd.isna(ema_proximity) else None,
                adx=adx_val
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return TechnicalIndicators()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate ADX (Average Directional Index)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Directional Movement
            up = high - high.shift()
            down = low.shift() - low
            
            pos_dm = up.where((up > down) & (up > 0), 0)
            neg_dm = down.where((down > up) & (down > 0), 0)
            
            pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
            
            # ADX
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
        except:
            return None
    
    def _calculate_stm_angle(self, candles: List[CandleData]) -> float:
        """Calculate EMA 6 angle for STM filter"""
        try:
            df = pd.DataFrame([{'close': c.close} for c in candles])
            ema_6 = df['close'].ewm(span=6, adjust=False).mean()
            
            if len(ema_6) >= 3:
                ema_change = ema_6.iloc[-1] - ema_6.iloc[-3]
                angle = np.degrees(np.arctan(ema_change / ema_6.iloc[-3]))
                return float(angle)
            return 0.0
        except:
            return 0.0
    
    def _all_candles_bullish(self, candles: List[CandleData]) -> bool:
        """Check if ALL candles are bullish (close > open)"""
        return len(candles) >= 3 and all(c.close > c.open for c in candles)
    
    def _all_candles_bearish(self, candles: List[CandleData]) -> bool:
        """Check if ALL candles are bearish (close < open)"""
        return len(candles) >= 3 and all(c.close < c.open for c in candles)
    
    def _get_adx_threshold(self, timeframe: str) -> float:
        """Get ADX threshold based on timeframe"""
        # 30s, 1m: ADX > 20
        if timeframe in ['30', '60']:
            return 20.0
        # 2m, 3m, 5m, 10m, 15m: ADX > 25
        else:
            return 25.0
    
    def analyze_flash_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Flash Mode Strategy with 3-tier urgency:
        
        NICE (basic):
        - 3 consecutive candles
        - RSI > 50 / < 50
        - Distance to BB ≥ 40%
        - EMAs close (≤ 0.5%)
        
        SHARP (nice + extras):
        - All NICE conditions
        - STM Filter (angle ≥ 15° or ≤ -15°)
        - ADX above threshold
        
        HOT (sharp + perfect):
        - All SHARP conditions
        - Distance to BB ≥ 50% (extra room)
        - EMA proximity ≤ 0.3% (very tight)
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close
        
        if not (indicators.bb_upper and indicators.bb_lower and indicators.bb_middle):
            return None
        
        bb_range = indicators.bb_upper - indicators.bb_lower
        stm_angle = self._calculate_stm_angle(candles)
        adx_threshold = self._get_adx_threshold(timeframe)
        
        # BUY SIGNAL
        if (self._all_candles_bullish(candles[-3:]) and
            indicators.rsi and indicators.rsi > 50 and
            indicators.ema_proximity and indicators.ema_proximity <= 0.5):
            
            distance_to_upper = indicators.bb_upper - current_price
            distance_pct = (distance_to_upper / bb_range) * 100
            
            # Must have at least 40% room
            if distance_pct >= 40:
                
                # Determine urgency level
                has_stm = stm_angle >= 15.0
                has_adx = indicators.adx and indicators.adx > adx_threshold
                has_extra_room = distance_pct >= 50
                has_tight_ema = indicators.ema_proximity <= 0.3
                
                # Urgency calculation
                if has_stm and has_adx and has_extra_room and has_tight_ema:
                    urgency = "hot"
                elif has_stm and has_adx:
                    urgency = "sharp"
                else:
                    urgency = "nice"
                
                reasons = [
                    "3 consecutive bullish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"Distance to upper BB: {distance_pct:.0f}%",
                    f"EMAs close (gap: {indicators.ema_proximity:.2f}%)"
                ]
                
                if has_stm:
                    reasons.append(f"STM: EMA angle {stm_angle:.1f}°")
                if has_adx:
                    reasons.append(f"ADX: {indicators.adx:.1f}")
                
                logger.info(f"✅ BUY ({urgency.upper()}): {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"STM={stm_angle:.1f}° | ADX={indicators.adx}")
                
                return SignalData(
                    asset=asset,
                    direction="buy",
                    timeframe=timeframe,
                    urgency=urgency,
                    confidence=min(95, 60 + (distance_pct * 0.5) + (10 if has_stm else 0) + (10 if has_adx else 0)),
                    reasons=reasons,
                    mode="flash",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        # SELL SIGNAL
        if (self._all_candles_bearish(candles[-3:]) and
            indicators.rsi and indicators.rsi < 50 and
            indicators.ema_proximity and indicators.ema_proximity <= 0.5):
            
            distance_to_lower = current_price - indicators.bb_lower
            distance_pct = (distance_to_lower / bb_range) * 100
            
            if distance_pct >= 40:
                
                has_stm = stm_angle <= -15.0
                has_adx = indicators.adx and indicators.adx > adx_threshold
                has_extra_room = distance_pct >= 50
                has_tight_ema = indicators.ema_proximity <= 0.3
                
                if has_stm and has_adx and has_extra_room and has_tight_ema:
                    urgency = "hot"
                elif has_stm and has_adx:
                    urgency = "sharp"
                else:
                    urgency = "nice"
                
                reasons = [
                    "3 consecutive bearish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"Distance to lower BB: {distance_pct:.0f}%",
                    f"EMAs close (gap: {indicators.ema_proximity:.2f}%)"
                ]
                
                if has_stm:
                    reasons.append(f"STM: EMA angle {stm_angle:.1f}°")
                if has_adx:
                    reasons.append(f"ADX: {indicators.adx:.1f}")
                
                logger.info(f"✅ SELL ({urgency.upper()}): {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"STM={stm_angle:.1f}° | ADX={indicators.adx}")
                
                return SignalData(
                    asset=asset,
                    direction="sell",
                    timeframe=timeframe,
                    urgency=urgency,
                    confidence=min(95, 60 + (distance_pct * 0.5) + (10 if has_stm else 0) + (10 if has_adx else 0)),
                    reasons=reasons,
                    mode="flash",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        return None
    
    def analyze_super_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """
        Super Mode Strategy:
        - All Flash conditions
        - STM Filter REQUIRED
        - ADX REQUIRED
        - Stricter thresholds
        - NO urgency displayed (urgency=None)
        """
        if len(candles) < 50:
            return None
        
        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close
        
        if not (indicators.bb_upper and indicators.bb_lower and indicators.bb_middle):
            return None
        
        bb_range = indicators.bb_upper - indicators.bb_lower
        stm_angle = self._calculate_stm_angle(candles)
        adx_threshold = self._get_adx_threshold(timeframe)
        
        # BUY SIGNAL
        if (self._all_candles_bullish(candles[-3:]) and
            indicators.rsi and indicators.rsi > 50 and
            indicators.ema_proximity and indicators.ema_proximity <= 0.3 and
            stm_angle >= 15.0 and
            indicators.adx and indicators.adx > adx_threshold):
            
            distance_to_upper = indicators.bb_upper - current_price
            distance_pct = (distance_to_upper / bb_range) * 100
            
            if distance_pct >= 40:
                reasons = [
                    "3 consecutive bullish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"Distance to upper BB: {distance_pct:.0f}%",
                    f"EMAs touching (gap: {indicators.ema_proximity:.2f}%)",
                    f"STM: EMA angle {stm_angle:.1f}°",
                    f"ADX: {indicators.adx:.1f}"
                ]
                
                logger.info(f"🔥 SUPER BUY: {asset} {timeframe} | RSI={indicators.rsi:.1f} | "
                           f"STM={stm_angle:.1f}° | ADX={indicators.adx:.1f}")
                
                return SignalData(
                    asset=asset,
                    direction="buy",
                    timeframe=timeframe,
                    urgency=None,  # No urgency in Super mode
                    confidence=min(98, 75 + (distance_pct * 0.4)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        # SELL SIGNAL
        if (self._all_candles_bearish(candles[-3:]) and
            indicators.rsi and indicators.rsi < 50 and
            indicators.ema_proximity and indicators.ema_proximity <= 0.3 and
            stm_angle <= -15.0 and
            indicators.adx and indicators.adx > adx_threshold):
            
            distance_to_lower = current_price - indicators.bb_lower
            distance_pct = (distance_to_lower / bb_range) * 100
            
            if distance_pct >= 40:
                reasons = [
                    "3 consecutive bearish candles",
                    f"RSI: {indicators.rsi:.1f}",
                    f"Distance to lower BB: {distance_pct:.0f}%",
                    f"EMAs touching (gap: {indicators.ema_proximity:.2f}%)",
                    f"STM: EMA angle {stm_angle:.1f}°",
                    f"ADX: {indicators.adx:.1f}"
                ]
                
                logger.info(f"🔥 SUPER SELL: {asset} {timeframe} | RSI={indicators.rsi:.1f} | "
                           f"STM={stm_angle:.1f}° | ADX={indicators.adx:.1f}")
                
                return SignalData(
                    asset=asset,
                    direction="sell",
                    timeframe=timeframe,
                    urgency=None,  # No urgency in Super mode
                    confidence=min(98, 75 + (distance_pct * 0.4)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )
        
        return None
