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
    
    # Trading assets to scan - API names (left side)
    ASSETS = [
        # Major Forex Pairs
        "EURUSD", "EURUSD_otc", "GBPUSD", "GBPUSD_otc", "USDJPY", "USDJPY_otc",
        "USDCHF", "USDCHF_otc", "USDCAD", "USDCAD_otc", "AUDUSD", "AUDUSD_otc",
        "AUDCAD", "AUDCAD_otc", "AUDCHF", "AUDCHF_otc", "AUDJPY", "AUDJPY_otc",
        "AUDNZD", "AUDNZD_otc", "CADCHF", "CADCHF_otc", "CADJPY", "CADJPY_otc",
        "CHFJPY", "CHFJPY_otc", "CHFNOK", "CHFNOK_otc", "EURAUD", "EURAUD_otc",
        "EURCAD", "EURCAD_otc", "EURCHF", "EURCHF_otc", "EURGBP", "EURGBP_otc",
        "EURJPY", "EURJPY_otc", "EURNZD", "EURNZD_otc", "GBPAUD", "GBPAUD_otc",
        "GBPCAD", "GBPCAD_otc", "GBPCHF", "GBPCHF_otc", "GBPJPY", "GBPJPY_otc",
        "NZDJPY", "NZDJPY_otc", "NZDUSD", "NZDUSD_otc",
        
        # Exotic Pairs (OTC only)
        "AEDCNY_otc", "BHDCNY_otc", "EURHUF_otc", "EURRUB_otc", "EURTRY_otc",
        "IRRUSD_otc", "KESUSD_otc", "NGNUSD_otc", "UAHUSD_otc", "ZARUSD_otc",
        "JODCNY_otc", "LBPUSD_otc", "MADUSD_otc", "OMRCNY_otc", "QARCNY_otc",
        "SARCNY_otc", "SYPUSD_otc", "TNDUSD_otc", "USDARS_otc", "USDBDT_otc",
        "USDBRL_otc", "USDCLP_otc", "USDCNH_otc", "USDCOP_otc", "USDDZD_otc",
        "USDEGP_otc", "USDIDR_otc", "USDINR_otc", "USDMXN_otc", "USDMYR_otc",
        "USDPHP_otc", "USDPKR_otc", "USDRUB_otc", "USDSGD_otc", "USDTHB_otc",
        "USDVND_otc", "YERUSD_otc",
        
        # Stocks
        "AAPL", "AAPL_otc", "AMZN_otc", "BABA", "BABA_otc", "CITI", "CITI_otc",
        "FDX_otc", "NFLX", "NFLX_otc", "TSLA", "TSLA_otc", "AMD_otc", "MARA_otc",
        "PLTR_otc", "GME_otc", "COIN_otc", "VIX_otc", "TWITTER", "TWITTER_otc",
        "VISA_otc", "AXP", "AXP_otc", "BA", "BA_otc", "CSCO", "CSCO_otc",
        "FB", "FB_otc", "INTC", "INTC_otc", "JNJ", "JNJ_otc", "JPM",
        "MCD", "MCD_otc", "MSFT", "MSFT_otc", "PFE", "PFE_otc", "XOM", "XOM_otc",
        
        # Cryptocurrencies
        "BTCGBP", "BTCJPY", "BTCUSD", "BTCUSD_otc", "ETHUSD", "ETHUSD_otc",
        "ADA-USD_otc", "AVAX_otc", "BCHEUR", "BCHGBP", "BCHJPY", "BITB_otc",
        "BNB-USD_otc", "DASH_USD", "DOGE_otc", "DOTUSD_otc", "LINK_otc",
        "LNKUSD", "LTCUSD_otc", "MATIC_otc", "SOL-USD_otc", "TON-USD_otc",
        "TRX-USD_otc", "XRPUSD_otc",
        
        # Commodities & Precious Metals
        "UKBrent", "UKBrent_otc", "USCrude", "USCrude_otc", "XNGUSD", "XNGUSD_otc",
        "XAGEUR", "XAGUSD", "XAGUSD_otc", "XAUEUR", "XAUUSD", "XAUUSD_otc",
        "XPDUSD", "XPDUSD_otc", "XPTUSD", "XPTUSD_otc",
        
        # Indices
        "AEX25", "AUS200", "AUS200_otc", "CAC40", "D30EUR", "D30EUR_otc",
        "DJI30", "DJI30_otc", "E35EUR", "E35EUR_otc", "E50EUR", "E50EUR_otc",
        "F40EUR", "F40EUR_otc", "H33HKD", "100GBP", "100GBP_otc", "JPN225",
        "JPN225_otc", "NASUSD", "NASUSD_otc", "SMI20", "SP500", "SP500_otc"
    ]
    
    # Display name mapping (API name → Frontend display name)
    ASSET_DISPLAY_NAMES = {
        # Forex - OTC
        "EURUSD_otc": "EURUSD OTC", "GBPUSD_otc": "GBPUSD OTC", "USDJPY_otc": "USDJPY OTC",
        "USDCHF_otc": "USDCHF OTC", "USDCAD_otc": "USDCAD OTC", "AUDUSD_otc": "AUDUSD OTC",
        "AUDCAD_otc": "AUDCAD OTC", "AUDCHF_otc": "AUDCHF OTC", "AUDJPY_otc": "AUDJPY OTC",
        "AUDNZD_otc": "AUDNZD OTC", "CADCHF_otc": "CADCHF OTC", "CADJPY_otc": "CADJPY OTC",
        "CHFJPY_otc": "CHFJPY OTC", "CHFNOK_otc": "CHFNOK OTC", "EURAUD_otc": "EURAUD OTC",
        "EURCAD_otc": "EURCAD OTC", "EURCHF_otc": "EURCHF OTC", "EURGBP_otc": "EURGBP OTC",
        "EURJPY_otc": "EURJPY OTC", "EURNZD_otc": "EURNZD OTC", "GBPAUD_otc": "GBPAUD OTC",
        "GBPCAD_otc": "GBPCAD OTC", "GBPCHF_otc": "GBPCHF OTC", "GBPJPY_otc": "GBPJPY OTC",
        "NZDJPY_otc": "NZDJPY OTC", "NZDUSD_otc": "NZDUSD OTC",
        
        # Exotic Pairs
        "AEDCNY_otc": "AEDCNY OTC", "BHDCNY_otc": "BHDCNY OTC", "EURHUF_otc": "EURHUF OTC",
        "EURRUB_otc": "EURRUB OTC", "EURTRY_otc": "EURTRY OTC", "IRRUSD_otc": "IRRUSD OTC",
        "KESUSD_otc": "KESUSD OTC", "NGNUSD_otc": "NGNUSD OTC", "UAHUSD_otc": "UAHUSD OTC",
        "ZARUSD_otc": "ZARUSD OTC", "JODCNY_otc": "JODCNY OTC", "LBPUSD_otc": "LBPUSD OTC",
        "MADUSD_otc": "MADUSD OTC", "OMRCNY_otc": "OMRCNY OTC", "QARCNY_otc": "QARCNY OTC",
        "SARCNY_otc": "SARCNY OTC", "SYPUSD_otc": "SYPUSD OTC", "TNDUSD_otc": "TNDUSD OTC",
        "USDARS_otc": "USDARS OTC", "USDBDT_otc": "USDBDT OTC", "USDBRL_otc": "USDBRL OTC",
        "USDCLP_otc": "USDCLP OTC", "USDCNH_otc": "USDCNH OTC", "USDCOP_otc": "USDCOP OTC",
        "USDDZD_otc": "USDDZD OTC", "USDEGP_otc": "USDEGP OTC", "USDIDR_otc": "USDIDR OTC",
        "USDINR_otc": "USDINR OTC", "USDMXN_otc": "USDMXN OTC", "USDMYR_otc": "USDMYR OTC",
        "USDPHP_otc": "USDPHP OTC", "USDPKR_otc": "USDPKR OTC", "USDRUB_otc": "USDRUB OTC",
        "USDSGD_otc": "USDSGD OTC", "USDTHB_otc": "USDTHB OTC", "USDVND_otc": "USDVND OTC",
        "YERUSD_otc": "YERUSD OTC",
        
        # Stocks
        "AAPL": "Apple", "AAPL_otc": "Apple OTC", "AMZN_otc": "Amazon OTC",
        "BABA": "Alibaba", "BABA_otc": "Alibaba OTC", "CITI": "Citigroup",
        "CITI_otc": "Citigroup OTC", "FDX_otc": "FedEx OTC", "NFLX": "Netflix",
        "NFLX_otc": "Netflix OTC", "TSLA": "Tesla", "TSLA_otc": "Tesla OTC",
        "AMD_otc": "Advanced Micro Devices OTC", "MARA_otc": "Marathon Digital OTC",
        "PLTR_otc": "Palantir Technologies OTC", "GME_otc": "GameStop Corp OTC",
        "COIN_otc": "Coinbase Global OTC", "VIX_otc": "VIX OTC", "TWITTER": "Twitter",
        "TWITTER_otc": "Twitter OTC", "VISA_otc": "Visa OTC", "AXP": "American Express",
        "AXP_otc": "American Express OTC", "BA": "Boeing", "BA_otc": "Boeing OTC",
        "CSCO": "Cisco Systems", "CSCO_otc": "Cisco Systems OTC", "FB": "Facebook Inc",
        "FB_otc": "Facebook Inc OTC", "INTC": "Intel", "INTC_otc": "Intel OTC",
        "JNJ": "Johnson & Johnson", "JNJ_otc": "Johnson & Johnson OTC",
        "JPM": "JPMorgan Chase", "MCD": "McDonald's", "MCD_otc": "McDonald's OTC",
        "MSFT": "Microsoft", "MSFT_otc": "Microsoft OTC", "PFE": "Pfizer",
        "PFE_otc": "Pfizer OTC", "XOM": "ExxonMobil", "XOM_otc": "ExxonMobil OTC",
        
        # Cryptocurrencies
        "BTCUSD_otc": "Bitcoin OTC", "ETHUSD_otc": "Ethereum OTC",
        "ADA-USD_otc": "Cardano OTC", "AVAX_otc": "Avalanche OTC",
        "BITB_otc": "Bitcoin ETF OTC", "BNB-USD_otc": "BNB OTC",
        "DASH_USD": "Dash", "DOGE_otc": "Dogecoin OTC", "DOTUSD_otc": "Polkadot OTC",
        "LINK_otc": "Chainlink OTC", "LNKUSD": "Chainlink", "LTCUSD_otc": "Litecoin OTC",
        "MATIC_otc": "Polygon OTC", "SOL-USD_otc": "Solana OTC",
        "TON-USD_otc": "Toncoin OTC", "TRX-USD_otc": "Tron OTC",
        "XRPUSD_otc": "Ripple OTC",
        
        # Commodities & Precious Metals
        "UKBrent": "Brent", "UKBrent_otc": "Brent Oil OTC", "USCrude": "WTI Crude Oil",
        "USCrude_otc": "WTI Crude Oil OTC", "XNGUSD": "Natural Gas",
        "XNGUSD_otc": "Natural Gas OTC", "XAGUSD": "Silver", "XAGUSD_otc": "Silver OTC",
        "XAUUSD": "Gold", "XAUUSD_otc": "Gold OTC", "XPDUSD": "Palladium",
        "XPDUSD_otc": "Palladium Spot OTC", "XPTUSD": "Platinum",
        "XPTUSD_otc": "Platinum Spot OTC",
        
        # Indices
        "AUS200_otc": "AUS200 OTC", "D30EUR_otc": "D30EUR OTC", "DJI30_otc": "DJI30 OTC",
        "E35EUR_otc": "E35EUR OTC", "E50EUR_otc": "E50EUR OTC", "F40EUR_otc": "F40EUR OTC",
        "H33HKD": "Hong Kong 33", "100GBP_otc": "100GBP OTC", "JPN225_otc": "JPN225 OTC",
        "NASUSD_otc": "NASUSD OTC", "SP500_otc": "SP500 OTC"
    }
    
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
        logger.info(f"📦 Created {len(batches)} batches from {len(self.ASSETS)} assets")
        return batches
    
    def get_display_name(self, api_asset_name: str) -> str:
        """Get frontend display name for asset"""
        return self.ASSET_DISPLAY_NAMES.get(api_asset_name, api_asset_name)
    
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
            
            logger.info(f"🟢 Scanner started - Mode: {mode}, Timeframe: {timeframe}, Total assets: {len(self.ASSETS)}, Batches: {len(self.asset_batches)}")
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
            self.current_batch_index = 0  # Reset batch index
            
            logger.info("🔴 Scanner stopped")
            return "Scanner stopped successfully"
            
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}")
            return f"Error stopping scanner: {e}"
    
    def _schedule_scanning(self, timeframe: str):
        """Schedule scanning based on timeframe requirements"""
        
        if timeframe in ["30", "60", "120"]:  # 30s, 1m, 2m - every 6 seconds
            self.scheduler.add_job(
                self._scan_next_batch,
                'interval',
                seconds=6,
                id=f'scan_{timeframe}',
                replace_existing=True
            )
            
        elif timeframe in ["180", "300", "600", "900"]:  # 3m, 5m, 10m, 15m
            # Get minute ranges based on timeframe
            minute_ranges = self._get_minute_ranges(timeframe)
            
            # Schedule batch scanning at specific minutes
            for start_min, end_min in minute_ranges:
                self.scheduler.add_job(
                    self._scan_all_batches_sequentially,
                    CronTrigger(minute=f"{start_min}-{end_min}"),
                    id=f'scan_{timeframe}_{start_min}_{end_min}',
                    replace_existing=True
                )
    
    def _get_minute_ranges(self, timeframe: str) -> List[tuple]:
        """Get minute ranges for timeframe"""
        if timeframe == "180":  # 3m
            return [
                (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17),
                (19, 20), (22, 23), (25, 26), (28, 29), (31, 32), (34, 35),
                (37, 38), (40, 41), (43, 44), (46, 47), (49, 50), (52, 53),
                (55, 56), (58, 59)
            ]
        elif timeframe == "300":  # 5m
            return [
                (3, 4), (8, 9), (13, 14), (18, 19), (23, 24), (28, 29),
                (33, 34), (38, 39), (43, 44), (48, 49), (53, 54), (58, 59)
            ]
        elif timeframe == "600":  # 10m
            return [(8, 9), (18, 19), (28, 29), (38, 39), (48, 49), (58, 59)]
        elif timeframe == "900":  # 15m
            return [(13, 14), (28, 29), (43, 44), (58, 59)]
        return []
    
    async def _scan_next_batch(self):
        """Scan next batch in rotation (for fast timeframes 30s, 60s, 120s)"""
        if not self.api_client or not self.is_scanning:
            return
        
        try:
            # Get current batch and rotate
            current_batch = self.asset_batches[self.current_batch_index]
            batch_num = self.current_batch_index + 1
            self.current_batch_index = (self.current_batch_index + 1) % len(self.asset_batches)
            
            logger.info(f"🔍 Scanning batch {batch_num}/{len(self.asset_batches)} ({len(current_batch)} assets)")
            
            await self._scan_batch(current_batch)
            
        except Exception as e:
            logger.error(f"❌ Batch scan error: {e}")
    
    async def _scan_all_batches_sequentially(self):
        """Scan all batches sequentially with 6-second delay (for 3m, 5m, 10m, 15m)"""
        if not self.api_client or not self.is_scanning:
            return
        
        try:
            logger.info(f"📊 Starting sequential scan of all {len(self.asset_batches)} batches")
            
            for batch_index, batch in enumerate(self.asset_batches):
                if not self.is_scanning:  # Check if still scanning
                    break
                
                logger.info(f"🔍 Scanning batch {batch_index + 1}/{len(self.asset_batches)} ({len(batch)} assets)")
                
                await self._scan_batch(batch)
                
                # Wait 6 seconds before next batch (except for last batch)
                if batch_index < len(self.asset_batches) - 1:
                    await asyncio.sleep(6)
            
            logger.info(f"✅ Completed sequential scan of all batches")
            
        except Exception as e:
            logger.error(f"❌ Sequential batch scan error: {e}")
    
    async def _scan_batch(self, batch: List[str]):
        """Scan a single batch of assets"""
        try:
            # Scan assets concurrently (max 5 at a time to avoid API limits)
            semaphore = asyncio.Semaphore(5)
            tasks = [self._scan_asset(asset, semaphore) for asset in batch]
            
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
                
                # Get display name for frontend
                display_name = self.get_display_name(asset)
                
                # Analyze based on current mode
                if self.current_mode == "flash":
                    signal = self.signal_engine.analyze_flash_mode(candles, self.current_timeframe, display_name)
                else:  # super
                    signal = self.signal_engine.analyze_super_mode(candles, self.current_timeframe, display_name)
                
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
