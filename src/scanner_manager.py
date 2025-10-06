"""
Scanner Manager for Zion Stryker Trading Bot
Handles scanning EURUSD_otc and automatic trade execution
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .models import SignalData, CandleData, TradeResult
from .signal_engine import SignalEngine
from .firebase_service import FirebaseService

try:
    from BinaryOptionsToolsV2.pocketoption.asyncronous import PocketOptionAsync
    USING_REAL_API = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("✅ Real Pocket Option API loaded")
except ImportError:
    from .mock_pocketoption import MockPocketOptionAsync as PocketOptionAsync
    USING_REAL_API = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("⚠️  Using MOCK Pocket Option API - No real trades will be executed!")

logger = logging.getLogger(__name__)

class ScannerManager:
    """Manages scanning operations with auto-trading for EURUSD_otc"""
    
    # Timeframe mapping: frontend -> backend scan interval
    TIMEFRAME_MAPPING = {
        "30": "30",    # 30s button -> 30s scan
        "60": "15",    # 1m button -> 15s scan
        "120": "60",   # 2m button -> 1m scan
        "180": "120",  # 3m button -> 2m scan
        "300": "180",  # 5m button -> 3m scan
        "600": "300",  # 10m button -> 5m scan
        "900": "600"   # 15m button -> 10m scan
    }
    
    def __init__(self, signal_engine: SignalEngine, firebase_service: FirebaseService):
        self.signal_engine = signal_engine
        self.firebase_service = firebase_service
        self.scheduler = AsyncIOScheduler()
        self.is_scanning = False
        self.current_mode = "flash"
        self.current_timeframe = "30"
        self.scan_timeframe = "30"  # Actual scan interval
        self.live_signals: List[SignalData] = []
        self.signal_history: List[SignalData] = []
        self.trade_history: List[TradeResult] = []
        self.api_client: Optional[PocketOptionAsync] = None
        self.signal_callback: Optional[Callable] = None
        self.auto_trade_enabled = True
        self.trade_amount = 1.0  # $1 per trade
    
    def set_signal_callback(self, callback: Callable):
        """Set callback for signal notifications"""
        self.signal_callback = callback
    
    async def start_scanner(self, mode: str, timeframe: str) -> str:
        """Start scanning with specified mode and timeframe"""
        try:
            if self.is_scanning:
                await self.stop_scanner()
            
            self.current_mode = mode
            self.current_timeframe = timeframe
            
            # Map frontend timeframe to backend scan interval
            self.scan_timeframe = self.TIMEFRAME_MAPPING.get(timeframe, timeframe)
            
            # Initialize Pocket Option API
            ssid = os.getenv("POCKET_OPTION_SSID", "demo_ssid")
            self.api_client = PocketOptionAsync(ssid)
            await asyncio.sleep(1)  # Connection stabilization
            
            # Configure scanning schedule based on scan timeframe
            self._schedule_scanning()
            
            self.scheduler.start()
            self.is_scanning = True
            
            logger.info(f"🟢 Scanner started - Mode: {mode}, Frontend TF: {timeframe}s, Scan TF: {self.scan_timeframe}s")
            return f"Scanner started in {mode} mode (frontend: {timeframe}s, scan: {self.scan_timeframe}s)"
            
        except Exception as e:
            logger.error(f"❌ Error starting scanner: {e}")
            raise
    
    async def stop_scanner(self) -> str:
        """Stop all scanning operations"""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            
            self.is_scanning = False
            self.api_client = None
            
            logger.info("🔴 Scanner stopped")
            return "Scanner stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}")
            return f"Error stopping scanner: {e}"
    
    def _schedule_scanning(self):
        """Schedule scanning based on scan timeframe"""
        scan_interval = int(self.scan_timeframe)
        
        # For short timeframes (15s, 30s, 1m), scan every interval
        if scan_interval <= 60:
            self.scheduler.add_job(
                self._scan_asset,
                'interval',
                seconds=scan_interval,
                id='scan_eurusd',
                replace_existing=True
            )
        else:
            # For longer timeframes, use interval scheduling
            self.scheduler.add_job(
                self._scan_asset,
                'interval',
                seconds=scan_interval,
                id='scan_eurusd',
                replace_existing=True
            )
    
    async def _scan_asset(self):
        """Scan EURUSD_otc for trading signals"""
        if not self.api_client or not self.is_scanning:
            return
        
        try:
            asset = "EURUSD_otc"
            scan_interval = int(self.scan_timeframe)
            
            # Get candle data (last 100 candles for analysis)
            candles_data = await self.api_client.get_candles(asset, scan_interval, 100)
            
            if not candles_data or len(candles_data) < 50:
                logger.warning(f"Insufficient candle data for {asset}")
                return
            
            # Convert to CandleData objects
            candles = [
                CandleData(
                    open=c['open'],
                    high=c['high'],
                    low=c['low'],
                    close=c['close'],
                    volume=c.get('volume', 0),
                    timestamp=datetime.fromtimestamp(c['timestamp'])
                )
                for c in candles_data
            ]
            
            # Analyze based on current mode
            signal = None
            if self.current_mode == "flash":
                signal = self.signal_engine.analyze_flash_mode(candles, self.scan_timeframe, asset)
            else:  # super
                signal = self.signal_engine.analyze_super_mode(candles, self.scan_timeframe, asset)
            
            if signal:
                await self._process_signal(signal)
                
        except Exception as e:
            logger.error(f"❌ Error scanning {asset}: {e}")
    
    async def _process_signal(self, signal: SignalData):
        """Process and execute trade based on signal"""
        try:
            # Add to live signals
            self.live_signals.append(signal)
            self.signal_history.append(signal)
            
            # Clean up expired live signals
            await self._cleanup_expired_signals()
            
            logger.info(f"📈 {signal.priority.upper()} Signal: {signal.asset} {signal.direction} ({signal.mode} mode, {signal.confidence:.1f}% confidence)")
            
            # Auto-execute trade
            if self.auto_trade_enabled and self.api_client:
                trade = await self._execute_trade(signal)
                if trade:
                    logger.info(f"💰 Trade executed: {trade.trade_id} - {trade.direction} {trade.asset} ${trade.amount}")
            
            # Send push notification
            try:
                await self.firebase_service.send_signal_notification(signal)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
            
            # Broadcast to connected clients
            if self.signal_callback:
                await self.signal_callback({
                    "type": "new_signal",
                    "signal": signal.dict(),
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _execute_trade(self, signal: SignalData) -> Optional[TradeResult]:
        """Execute trade based on signal"""
        try:
            # Convert frontend timeframe to expiry seconds
            expiry_seconds = int(self.current_timeframe)
            
            # Place trade
            trade = await self.api_client.place_trade(
                asset=signal.asset,
                direction=signal.direction,
                amount=self.trade_amount,
                expiry_seconds=expiry_seconds
            )
            
            # Update trade with mode and timeframe
            trade.mode = signal.mode
            trade.timeframe = self.current_timeframe
            
            # Store in trade history
            self.trade_history.append(trade)
            
            # Broadcast trade to clients
            if self.signal_callback:
                await self.signal_callback({
                    "type": "new_trade",
                    "trade": trade.dict(),
                    "timestamp": datetime.now().isoformat()
                })
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return None
    
    async def _cleanup_expired_signals(self):
        """Remove expired signals from live list"""
        try:
            # Keep signals for 2x the timeframe duration
            expiry_minutes = int(self.current_timeframe) / 60 * 2
            cutoff_time = datetime.now() - timedelta(minutes=expiry_minutes)
            
            self.live_signals = [
                s for s in self.live_signals 
                if s.timestamp > cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error cleaning up signals: {e}")
    
    def get_live_signals(self) -> List[SignalData]:
        """Get current live signals"""
        return self.live_signals
    
    def get_signal_history(self, limit: int = 50) -> List[SignalData]:
        """Get signal history"""
        return self.signal_history[-limit:]
    
    def get_trade_history(self, limit: int = 50) -> List[TradeResult]:
        """Get trade history"""
        return self.trade_history[-limit:]
    
    async def get_trade_results(self) -> dict:
        """Get trade statistics"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "pending": 0,
                "win_rate": 0.0,
                "total_profit": 0.0
            }
        
        # Update trade results from API
        if self.api_client:
            try:
                all_trades = await self.api_client.get_all_trades()
                # Update local trade history with results
                trade_map = {t.trade_id: t for t in all_trades}
                for i, trade in enumerate(self.trade_history):
                    if trade.trade_id in trade_map:
                        self.trade_history[i] = trade_map[trade.trade_id]
            except Exception as e:
                logger.warning(f"Failed to update trade results: {e}")
        
        wins = sum(1 for t in self.trade_history if t.result == "win")
        losses = sum(1 for t in self.trade_history if t.result == "loss")
        pending = sum(1 for t in self.trade_history if t.result == "pending")
        total_profit = sum(t.profit or 0 for t in self.trade_history)
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
        
        return {
            "total_trades": len(self.trade_history),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2)
        }
