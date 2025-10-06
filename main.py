"""
Zion Stryker Trading Bot - Main Application
Automated Binary Options Trading Bot for EURUSD_otc
"""
import os
import logging
import asyncio
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Import our custom modules
from src.signal_engine import SignalEngine
from src.scanner_manager import ScannerManager
from src.models import SignalResponse, ScannerToggleRequest, TradeResultsResponse
from src.firebase_service import FirebaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
scanner_manager: Optional[ScannerManager] = None
connected_clients: List[WebSocket] = []
keep_alive_task: Optional[asyncio.Task] = None

async def keep_alive_ping():
    """Self-ping to keep Railway service alive"""
    while True:
        try:
            await asyncio.sleep(300)  # Ping every 5 minutes
            
            # Get the Railway domain or use localhost
            domain = os.getenv("REPLIT_DEV_DOMAIN") or os.getenv("RAILWAY_STATIC_URL") or "localhost:5000"
            url = f"http://{domain}/api/health"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                logger.info(f"🏓 Keep-alive ping: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Keep-alive ping failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global scanner_manager, keep_alive_task
    
    # Startup
    logger.info("🚀 Starting Zion Stryker Trading Bot...")
    
    try:
        # Initialize Firebase service
        firebase_service = FirebaseService()
        
        # Initialize signal engine
        signal_engine = SignalEngine()
        
        # Initialize scanner manager
        scanner_manager = ScannerManager(signal_engine, firebase_service)
        scanner_manager.set_signal_callback(broadcast_to_clients)
        
        # Start keep-alive task
        keep_alive_task = asyncio.create_task(keep_alive_ping())
        
        # Check API mode
        from src.scanner_manager import USING_REAL_API
        if USING_REAL_API:
            logger.info("✅ All services initialized successfully - REAL TRADING MODE")
        else:
            logger.warning("⚠️  All services initialized - MOCK TRADING MODE (No real money)")
            logger.warning("⚠️  To enable real trading, see README for Pocket Option API setup")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Zion Stryker Trading Bot...")
    
    if scanner_manager:
        await scanner_manager.stop_scanner()
    
    if keep_alive_task:
        keep_alive_task.cancel()

app = FastAPI(
    title="Zion Stryker Trading Bot",
    description="Automated Binary Options Trading Bot for EURUSD_otc",
    version="3.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not connected_clients:
        return
    
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Zion Stryker Trading Bot",
        "version": "3.0.0",
        "status": "running",
        "asset": "EURUSD_otc",
        "modes": ["flash", "super"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    global scanner_manager
    
    return {
        "status": "healthy",
        "scanner_active": scanner_manager.is_scanning if scanner_manager else False,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/scanner/toggle")
async def toggle_scanner(request: dict):
    """Toggle scanner on/off with mode and timeframe selection"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    try:
        # Handle both old and new request formats
        if "enabled" in request:
            enabled = request["enabled"]
            action = "start" if enabled else "stop"
            mode = request.get("mode", "flash")
            timeframe = request.get("timeframe", "30")
        else:
            action = request.get("action", "start")
            mode = request.get("mode", "flash") 
            timeframe = request.get("timeframe", "30")
        
        if action == "start":
            result = await scanner_manager.start_scanner(
                mode=mode,
                timeframe=timeframe
            )
            
            # Broadcast to all connected clients
            await broadcast_to_clients({
                "type": "scanner_status",
                "status": "started", 
                "mode": mode,
                "timeframe": timeframe,
                "scan_timeframe": scanner_manager.scan_timeframe,
                "timestamp": datetime.now().isoformat()
            })
            
        else:  # stop
            result = await scanner_manager.stop_scanner()
            
            await broadcast_to_clients({
                "type": "scanner_status",
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            })
        
        return JSONResponse(content={
            "success": True,
            "action": action,
            "enabled": action == "start",
            "message": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error toggling scanner: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/live")
async def get_live_signals():
    """Get current live signals"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    signals = scanner_manager.get_live_signals()
    
    return SignalResponse(
        signals=signals,
        count=len(signals),
        timestamp=datetime.now()
    )

@app.get("/api/signals/history")
async def get_signal_history(limit: int = 50):
    """Get signal history"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    signals = scanner_manager.get_signal_history(limit)
    
    return SignalResponse(
        signals=signals,
        count=len(signals),
        timestamp=datetime.now()
    )

@app.get("/api/trades/history")
async def get_trade_history(limit: int = 50):
    """Get trade history"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    trades = scanner_manager.get_trade_history(limit)
    
    return {
        "trades": [t.dict() for t in trades],
        "count": len(trades),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/trades/results")
async def get_trade_results():
    """Get trade results and statistics"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    results = await scanner_manager.get_trade_results()
    
    return TradeResultsResponse(
        trades=scanner_manager.get_trade_history(),
        total_trades=results["total_trades"],
        wins=results["wins"],
        losses=results["losses"],
        pending=results["pending"],
        win_rate=results["win_rate"],
        total_profit=results["total_profit"]
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to Zion Stryker Trading Bot",
            "asset": "EURUSD_otc",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received from client: {data}")
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
