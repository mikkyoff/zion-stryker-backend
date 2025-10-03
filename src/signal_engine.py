"""
Signal Engine for Zion Stryker Trading Bot
Implements Flash and Super mode trading strategies
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
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
            df = pd.DataFrame([{\
                'close': c.close,\
                'high': c.high,\
                'low': c.low,\
                'open': c.open\
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

    def _check_ema_about_to_cross(self, candles: List[CandleData], direction: str) -> Tuple[bool, float]:
        """
        Check if EMAs are about to cross (very close but haven't crossed yet)

        For BUY: EMA 6 approaching from BELOW, gap shrinking
        For SELL: EMA 6 approaching from ABOVE, gap shrinking

        Returns: (is_about_to_cross, current_gap_percentage)
        """
        if len(candles) < 10:
            return False, 0.0

        try:
            df = pd.DataFrame([{'close': c.close} for c in candles])
            ema_6 = df['close'].ewm(span=6, adjust=False).mean()
            ema_18 = df['close'].ewm(span=18, adjust=False).mean()

            # Current values
            current_6 = ema_6.iloc[-1]
            current_18 = ema_18.iloc[-1]

            # Previous values (2 candles ago)
            prev_6 = ema_6.iloc[-3]
            prev_18 = ema_18.iloc[-3]

            # Calculate gaps
            current_gap = abs(current_6 - current_18)
            prev_gap = abs(prev_6 - prev_18)
            gap_pct = (current_gap / current_18) * 100

            # Check if gap is shrinking (EMAs getting closer)
            gap_shrinking = current_gap < prev_gap

            if direction == "buy":
                # EMA 6 must be BELOW EMA 18 (hasn't crossed)
                # AND gap must be very small (≤ 0.3%)
                # AND gap must be shrinking
                if current_6 < current_18 and gap_pct <= 0.3 and gap_shrinking:
                    return True, gap_pct

            elif direction == "sell":
                # EMA 6 must be ABOVE EMA 18 (hasn't crossed)
                # AND gap must be very small (≤ 0.3%)
                # AND gap must be shrinking
                if current_6 > current_18 and gap_pct <= 0.3 and gap_shrinking:
                    return True, gap_pct

            return False, gap_pct

        except Exception as e:
            logger.error(f"Error checking EMA cross: {e}")
            return False, 0.0

    def _all_candles_bullish(self, candles: List[CandleData]) -> bool:
        """Check if ALL last 3 candles are bullish (close > open)"""
        if len(candles) < 3:
            return False

        last_3 = candles[-3:]

        # DEBUG: Log each candle
        for i, c in enumerate(last_3, 1):
            is_bullish = c.close > c.open
            logger.debug(f"  Candle {i}: O={c.open:.5f} C={c.close:.5f} {'✅ BULL' if is_bullish else '❌ BEAR'}")

        all_bullish = all(c.close > c.open for c in last_3)
        logger.debug(f"  All 3 bullish? {all_bullish}")

        return all_bullish

    def _all_candles_bearish(self, candles: List[CandleData]) -> bool:
        """Check if ALL last 3 candles are bearish (close < open)"""
        if len(candles) < 3:
            return False

        last_3 = candles[-3:]

        # DEBUG: Log each candle
        for i, c in enumerate(last_3, 1):
            is_bearish = c.close < c.open
            logger.debug(f"  Candle {i}: O={c.open:.5f} C={c.close:.5f} {'✅ BEAR' if is_bearish else '❌ BULL'}")

        all_bearish = all(c.close < c.open for c in last_3)
        logger.debug(f"  All 3 bearish? {all_bearish}")

        return all_bearish

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
        Flash Mode Strategy with 3-tier urgency
        """
        if len(candles) < 50:
            return None

        # 🔥 ULTRA DEBUG LOGGING
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 ANALYZING {asset} at {timeframe}s timeframe")
        logger.info(f"{'='*60}")
        
        # Log last 3 candles RAW DATA
        logger.info(f"\n📊 LAST 3 CANDLES (RAW DATA):")
        for i, candle in enumerate(candles[-3:], 1):
            direction = "🟢 BULL" if candle.close > candle.open else "🔴 BEAR"
            logger.info(f"  Candle {i}: Open={candle.open:.5f}, Close={candle.close:.5f}, High={candle.high:.5f}, Low={candle.low:.5f} {direction}")

        indicators = self.calculate_indicators(candles, timeframe)
        current_price = candles[-1].close

        # Log calculated indicators
        logger.info(f"\n📈 CALCULATED INDICATORS:")
        logger.info(f"  Current Price: {current_price:.5f}")
        logger.info(f"  RSI: {indicators.rsi:.2f}" if indicators.rsi else "  RSI: None")
        logger.info(f"  BB Upper: {indicators.bb_upper:.5f}" if indicators.bb_upper else "  BB Upper: None")
        logger.info(f"  BB Middle: {indicators.bb_middle:.5f}" if indicators.bb_middle else "  BB Middle: None")
        logger.info(f"  BB Lower: {indicators.bb_lower:.5f}" if indicators.bb_lower else "  BB Lower: None")
        logger.info(f"  EMA 6: {indicators.ema_6:.5f}" if indicators.ema_6 else "  EMA 6: None")
        logger.info(f"  EMA 18: {indicators.ema_18:.5f}" if indicators.ema_18 else "  EMA 18: None")
        logger.info(f"  ADX: {indicators.adx:.2f}" if indicators.adx else "  ADX: None")

        if not (indicators.bb_upper and indicators.bb_lower and indicators.bb_middle):
            logger.warning(f"❌ Missing BB indicators for {asset}")
            return None

        bb_range = indicators.bb_upper - indicators.bb_lower
        stm_angle = self._calculate_stm_angle(candles)
        adx_threshold = self._get_adx_threshold(timeframe)

        # BUY SIGNAL CHECK
        logger.info(f"\n🔍 CHECKING BUY CONDITIONS:")
        
        candles_bullish = self._all_candles_bullish(candles)
        logger.info(f"  ✓ All 3 candles bullish? {candles_bullish}")
        
        rsi_check = indicators.rsi and indicators.rsi > 50
        logger.info(f"  ✓ RSI > 50? {rsi_check} (RSI={indicators.rsi:.1f})" if indicators.rsi else "  ✗ RSI is None")
        
        ema_about_to_cross_buy, ema_gap = self._check_ema_about_to_cross(candles, "buy")
        logger.info(f"  ✓ EMAs about to cross (buy)? {ema_about_to_cross_buy} (gap={ema_gap:.3f}%)")

        if (candles_bullish and
            indicators.rsi and indicators.rsi > 50 and
            ema_about_to_cross_buy):

            distance_to_upper = indicators.bb_upper - current_price
            distance_pct = (distance_to_upper / bb_range) * 100

            logger.info(f"  ✓ Distance to upper BB: {distance_pct:.1f}% (need ≥40%)")

            # Must have at least 40% room
            if distance_pct >= 40:

                # Determine urgency level
                has_stm = stm_angle >= 15.0
                has_adx = indicators.adx and indicators.adx > adx_threshold
                has_extra_room = distance_pct >= 50
                has_tight_ema = ema_gap <= 0.15

                # Urgency calculation
                if has_stm and has_adx and has_extra_room and has_tight_ema:
                    urgency = "hot"
                elif has_stm and has_adx:
                    urgency = "sharp"
                else:
                    urgency = "nice"

                reasons = [\
                    "3 consecutive bullish candles",\
                    f"RSI: {indicators.rsi:.1f}",\
                    f"Distance to upper BB: {distance_pct:.0f}%",\
                    f"EMAs about to cross (gap: {ema_gap:.2f}%)"\
                ]

                if has_stm:
                    reasons.append(f"ADX: {indicators.adx:.1f}")

                logger.info(f"✅ BUY ({urgency.upper()}): {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"EMA gap={ema_gap:.2f}% | STM={stm_angle:.1f}° | ADX={indicators.adx}")

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

        # SELL SIGNAL CHECK
        logger.info(f"\n🔍 CHECKING SELL CONDITIONS:")

        candles_bearish = self._all_candles_bearish(candles)
        logger.info(f"  ✓ All 3 candles bearish? {candles_bearish}")
        
        rsi_check_sell = indicators.rsi and indicators.rsi < 50
        logger.info(f"  ✓ RSI < 50? {rsi_check_sell} (RSI={indicators.rsi:.1f})" if indicators.rsi else "  ✗ RSI is None")
        
        ema_about_to_cross_sell, ema_gap = self._check_ema_about_to_cross(candles, "sell")
        logger.info(f"  ✓ EMAs about to cross (sell)? {ema_about_to_cross_sell} (gap={ema_gap:.3f}%)")

        if (candles_bearish and
            indicators.rsi and indicators.rsi < 50 and
            ema_about_to_cross_sell):

            distance_to_lower = current_price - indicators.bb_lower
            distance_pct = (distance_to_lower / bb_range) * 100

            logger.info(f"  ✓ Distance to lower BB: {distance_pct:.1f}% (need ≥40%)")

            if distance_pct >= 40:

                has_stm = stm_angle <= -15.0
                has_adx = indicators.adx and indicators.adx > adx_threshold
                has_extra_room = distance_pct >= 50
                has_tight_ema = ema_gap <= 0.15

                if has_stm and has_adx and has_extra_room and has_tight_ema:
                    urgency = "hot"
                elif has_stm and has_adx:
                    urgency = "sharp"
                else:
                    urgency = "nice"

                reasons = [\
                    "3 consecutive bearish candles",\
                    f"RSI: {indicators.rsi:.1f}",\
                    f"Distance to lower BB: {distance_pct:.0f}%",\
                    f"EMAs about to cross (gap: {ema_gap:.2f}%)"\
                ]

                if has_stm:
                    reasons.append(f"STM: EMA angle {stm_angle:.1f}°")
                if has_adx:
                    reasons.append(f"ADX: {indicators.adx:.1f}")

                logger.info(f"✅ SELL ({urgency.upper()}): {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"EMA gap={ema_gap:.2f}% | STM={stm_angle:.1f}° | ADX={indicators.adx}")

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

        logger.info(f"❌ No signal for {asset} {timeframe}")
        return None

    def analyze_super_mode(self, candles: List[CandleData], timeframe: str, asset: str) -> Optional[SignalData]:
        """Super Mode Strategy - same logic as Flash but stricter"""
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
        candles_bullish = self._all_candles_bullish(candles)
        ema_about_to_cross_buy, ema_gap = self._check_ema_about_to_cross(candles, "buy")

        if (candles_bullish and
            indicators.rsi and indicators.rsi > 50 and
            ema_about_to_cross_buy and
            ema_gap <= 0.15 and  # Very tight
            stm_angle >= 15.0 and
            indicators.adx and indicators.adx > adx_threshold):

            distance_to_upper = indicators.bb_upper - current_price
            distance_pct = (distance_to_upper / bb_range) * 100

            if distance_pct >= 50:  # Super needs MORE room

                reasons = [\
                    "3 consecutive bullish candles",\
                    f"RSI: {indicators.rsi:.1f}",\
                    f"Distance to upper BB: {distance_pct:.0f}%",\
                    f"EMAs about to cross (gap: {ema_gap:.2f}%)",\
                    f"STM: EMA angle {stm_angle:.1f}°",\
                    f"ADX: {indicators.adx:.1f}"\
                ]

                logger.info(f"✅ SUPER BUY: {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"EMA gap={ema_gap:.2f}% | STM={stm_angle:.1f}° | ADX={indicators.adx}")

                return SignalData(
                    asset=asset,
                    direction="buy",
                    timeframe=timeframe,
                    urgency="",  # No urgency for Super mode
                    confidence=min(98, 75 + (distance_pct * 0.4)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )

        # SELL SIGNAL
        candles_bearish = self._all_candles_bearish(candles)
        ema_about_to_cross_sell, ema_gap = self._check_ema_about_to_cross(candles, "sell")

        if (candles_bearish and
            indicators.rsi and indicators.rsi < 50 and
            ema_about_to_cross_sell and
            ema_gap <= 0.15 and
            stm_angle <= -15.0 and
            indicators.adx and indicators.adx > adx_threshold):

            distance_to_lower = current_price - indicators.bb_lower
            distance_pct = (distance_to_lower / bb_range) * 100

            if distance_pct >= 50:

                reasons = [\
                    "3 consecutive bearish candles",\
                    f"RSI: {indicators.rsi:.1f}",\
                    f"Distance to lower BB: {distance_pct:.0f}%",\
                    f"EMAs about to cross (gap: {ema_gap:.2f}%)",\
                    f"STM: EMA angle {stm_angle:.1f}°",\
                    f"ADX: {indicators.adx:.1f}"\
                ]

                logger.info(f"✅ SUPER SELL: {asset} {timeframe} | "
                           f"RSI={indicators.rsi:.1f} | BB={distance_pct:.0f}% | "
                           f"EMA gap={ema_gap:.2f}% | STM={stm_angle:.1f}° | ADX={indicators.adx}")

                return SignalData(
                    asset=asset,
                    direction="sell",
                    timeframe=timeframe,
                    urgency="",  # No urgency for Super mode
                    confidence=min(98, 75 + (distance_pct * 0.4)),
                    reasons=reasons,
                    mode="super",
                    timestamp=datetime.now(),
                    expiry=int(timeframe)
                )

        return None
