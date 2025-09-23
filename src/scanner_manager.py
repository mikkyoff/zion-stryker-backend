"""
Scanner Manager for Zion Stryker Trading Bot
Handles batch scanning with specific timing intervals per timeframe
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
try:
    from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
except ImportError:
    from .mock_pocketoption import MockPocketOptionAsync as PocketOptionAsync
import os

from .models import SignalData, CandleData
from .signal_engine import SignalEngine
from .firebase_service import FirebaseService

logger = logging.getLogger(__name__)

class ScannerManager:
    """Manages scanning operations with batch processing and timeframe-specific intervals"""
    
    # Trading assets to scan (25-asset batches)
    ASSETS = [
        "EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "USDCHF_otc", "USDCAD_otc",
        "AUDUSD_otc", "NZDUSD_otc", "EURGBP_otc", "EURJPY_otc", "GBPJPY_otc",
        "XAUUSD_otc", "BTCUSD_otc", "ETHUSD_otc", "LTCUSD_otc", "XRPUSD_otc",
        "ADAUSD_otc", "DOTUSD_otc", "UNIUSD_otc", "LINKUSD_otc", "SOLUSD_otc",
        "OIL_otc", "GAS_otc", "APPLE", "GOOGLE", "AMAZON"
    ]
    
    def __init__(self, signal_engine: SignalEngine, firebase_service: FirebaseService):
        self.signal_engine = signal_engine
        self.firebase_service = firebase_service
        self.scheduler = AsyncIOScheduler()
        self.is_scanning = False
        self.current_mode = "flash"
        self.current_timeframe = "60"
        self.live_signals: List[SignalData] = []
        self.signal_history: List[SignalData] = []
        self.api_client: Optional[PocketOptionAsync] = None
        self.asset_batches = self._create_asset_batches()
        self.current_batch_index = 0
        
        # Callbacks for real-time updates
        self.signal_callback: Optional[Callable] = None
        
    def _create_asset_batches(self) -> List[List[str]]:
        """Create 25-asset batches for load balancing"""
        batches = []
        for i in range(0, len(self.ASSETS), 25):
            batch = self.ASSETS[i:i+25]
            batches.append(batch)
        return batches
    
    async def start_scanner(self, mode: str, timeframe: str) -> str:
        """Start scanning with specified mode and timeframe"""
        try:
            if self.is_scanning:
                await self.stop_scanner()
            
            self.current_mode = mode
            self.current_timeframe = timeframe
            
            # Initialize Pocket Option API
            ssid = os.getenv("POCKET_OPTION_SSID")
            if not ssid:
                raise ValueError("POCKET_OPTION_SSID not configured")
            
            self.api_client = PocketOptionAsync(ssid)
            await asyncio.sleep(2)  # Connection stabilization
            
            # Configure scanning schedule based on timeframe
            self._schedule_scanning(timeframe)
            
            self.scheduler.start()
            self.is_scanning = True
            
            logger.info(f"🟢 Scanner started - Mode: {mode}, Timeframe: {timeframe}")
            return f"Scanner started in {mode} mode with {timeframe}s timeframe"
            
        except Exception as e:
            logger.error(f"❌ Failed to start scanner: {e}")
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
    
    def _schedule_scanning(self, timeframe: str):
        """Schedule scanning based on timeframe requirements"""
        
        if timeframe in ["30", "60", "120"]:  # 30s, 1m, 2m - every 6 seconds
            self.scheduler.add_job(
                self._scan_batch,
                'interval',
                seconds=6,
                id=f'scan_{timeframe}',
                replace_existing=True
            )
            
        elif timeframe == "180":  # 3m - specific minute windows
            minute_ranges = [
                (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17),
                (19, 20), (22, 23), (25, 26), (28, 29), (31, 32), (34, 35),
                (37, 38), (40, 41), (43, 44), (46, 47), (49, 50), (52, 53),
                (55, 56), (58, 59)
            ]
            for start_min, end_min in minute_ranges:
                self.scheduler.add_job(
                    self._scan_batch,
                    CronTrigger(minute=f"{start_min}-{end_min}"),
                    id=f'scan_3m_{start_min}_{end_min}',
                    replace_existing=True
                )
                
        elif timeframe == "300":  # 5m - specific minute windows
            minute_ranges = [
                (3, 4), (8, 9), (13, 14), (18, 19), (23, 24), (28, 29),
                (33, 34), (38, 39), (43, 44), (48, 49), (53, 54), (58, 59)
            ]
            for start_min, end_min in minute_ranges:
                self.scheduler.add_job(
                    self._scan_batch,
                    CronTrigger(minute=f"{start_min}-{end_min}"),
                    id=f'scan_5m_{start_min}_{end_min}',
                    replace_existing=True
                )
                
        elif timeframe == "600":  # 10m - specific minute windows
            minute_ranges = [(8, 9), (18, 19), (28, 29), (38, 39), (48, 49), (58, 59)]
            for start_min, end_min in minute_ranges:
                self.scheduler.add_job(
                    self._scan_batch,
                    CronTrigger(minute=f"{start_min}-{end_min}"),
                    id=f'scan_10m_{start_min}_{end_min}',
                    replace_existing=True
                )
                
        elif timeframe == "900":  # 15m - specific minute windows
            minute_ranges = [(13, 14), (28, 29), (43, 44), (58, 59)]
            for start_min, end_min in minute_ranges:
                self.scheduler.add_job(
                    self._scan_batch,
                    CronTrigger(minute=f"{start_min}-{end_min}"),
                    id=f'scan_15m_{start_min}_{end_min}',
                    replace_existing=True
                )
    
    async def _scan_batch(self):
        """Scan current batch of 25 assets"""
        if not self.api_client or not self.is_scanning:
            return
        
        try:
            # Get current batch (rotate through batches for load balancing)
            current_batch = self.asset_batches[self.current_batch_index]
            self.current_batch_index = (self.current_batch_index + 1) % len(self.asset_batches)
            
            logger.info(f"🔍 Scanning batch {self.current_batch_index + 1} with {len(current_batch)} assets")
            
            # Scan assets concurrently (max 5 at a time to avoid API limits)
            semaphore = asyncio.Semaphore(5)
            tasks = [self._scan_asset(asset, semaphore) for asset in current_batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            signals_found = 0
            for result in results:
                if isinstance(result, SignalData):
                    await self._process_signal(result)
                    signals_found += 1
                elif isinstance(result, Exception):
                    logger.warning(f"Asset scan error: {result}")
            
            if signals_found > 0:
                logger.info(f"✅ Found {signals_found} signals in batch")
                
        except Exception as e:
            logger.error(f"❌ Batch scan error: {e}")
    
    async def _scan_asset(self, asset: str, semaphore: asyncio.Semaphore) -> Optional[SignalData]:
        """Scan individual asset for trading signals"""
        async with semaphore:
            try:
                # Get candle data (last 100 candles for analysis)
                timeframe_int = int(self.current_timeframe)
                candles_data = await self.api_client.get_candles(asset, timeframe_int, 100)
                
                if not candles_data or len(candles_data) < 50:
                    return None
                
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
                if self.current_mode == "flash":
                    signal = self.signal_engine.analyze_flash_mode(candles, self.current_timeframe, asset)
                else:  # super
                    signal = self.signal_engine.analyze_super_mode(candles, self.current_timeframe, asset)
                
                return signal
                
            except Exception as e:
                logger.warning(f"Error scanning {asset}: {e}")
                return None
    
    async def _process_signal(self, signal: SignalData):
        """Process and store new trading signal"""
        try:
            # Add to live signals
            self.live_signals.append(signal)
            
            # Clean up expired live signals (based on timeframe)
            await self._cleanup_expired_signals()
            
            # Send push notification
            await self.firebase_service.send_signal_notification(signal)
            
            # Broadcast to connected clients
            if self.signal_callback:
                await self.signal_callback({
                    "type": "new_signal",
                    "signal": signal.dict(),
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"📈 {signal.priority.upper()} Signal: {signal.asset} {signal.direction} ({signal.mode} mode)")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _cleanup_expired_signals(self):
        """Move expired live signals to history"""
        timeframe_seconds = int(self.current_timeframe)
        expiry_time = datetime.now() - timedelta(seconds=timeframe_seconds)
        
        # Find expired signals
        expired_signals = [s for s in self.live_signals if s.timestamp < expiry_time]
        
        # Move to history
        for signal in expired_signals:
            self.signal_history.append(signal)
            self.live_signals.remove(signal)
        
        # Limit history to 100 signals (newest first)
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
    
    def get_live_signals(self) -> List[Dict]:
        """Get current live signals"""
        return [signal.dict() for signal in self.live_signals]
    
    def get_signal_history(self) -> List[Dict]:
        """Get signal history (newest first)"""
        return [signal.dict() for signal in reversed(self.signal_history)]
    
    def set_signal_callback(self, callback: Callable):
        """Set callback for real-time signal updates"""
        self.signal_callback = callback