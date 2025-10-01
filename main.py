"""
Zion Stryker Backend - Binary Options Trading Bot
FastAPI backend for sophisticated trading signal generation
"""

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
import logging
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from src.signal_engine import SignalEngine
from src.scanner_manager import ScannerManager
from src.models import SignalResponse, ScannerToggleRequest
from src.firebase_service import FirebaseService

app = FastAPI(
    title="Zion Stryker Trading Bot",
    description="Advanced Binary Options Trading Signal Engine",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
scanner_manager: Optional[ScannerManager] = None
connected_clients: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global scanner_manager
    logger.info("🚀 Starting Zion Stryker Backend...")

    try:
        firebase_service = FirebaseService()
        signal_engine = SignalEngine()
        scanner_manager = ScannerManager(signal_engine, firebase_service)
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global scanner_manager
    if scanner_manager:
        await scanner_manager.stop_scanner()
    logger.info("🔴 Zion Stryker Backend stopped")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Zion Stryker Trading Bot Backend",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "scanner_active": scanner_manager.is_scanning if scanner_manager else False
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "signal_engine": "running",
            "scanner_manager": "initialized" if scanner_manager else "not_initialized",
            "firebase": "configured",
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/scanner/toggle")
async def toggle_scanner(request: dict):
    """Toggle scanner on/off"""
    global scanner_manager

    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")

    try:
        # Handle both request formats
        if "enabled" in request:
            enabled = request["enabled"]
            action = "start" if enabled else "stop"
            mode = request.get("mode", "flash")
            timeframe = request.get("timeframe", "60")
        else:
            action = request.get("action", "start")
            mode = request.get("mode", "flash")
            timeframe = request.get("timeframe", "60")

        if action == "start":
            result = await scanner_manager.start_scanner(mode=mode, timeframe=timeframe)
            await broadcast_to_clients({
                "type": "scanner_status",
                "status": "started",
                "mode": mode,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            })
        else:
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time signals"""
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to Zion Stryker signals",
            "timestamp": datetime.now().isoformat()
        })

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

@app.get("/api/signals/live")
async def get_live_signals():
    """Get current live signals"""
    if not scanner_manager:
        return {"signals": []}

    return {
        "signals": scanner_manager.get_live_signals(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals/history")
async def get_signal_history():
    """Get signal history"""
    if not scanner_manager:
        return {"history": []}

    return {
        "history": scanner_manager.get_signal_history(),
        "timestamp": datetime.now().isoformat()
    }

async def broadcast_to_clients(message: dict):
    """Broadcast to all connected WebSocket clients"""
    if not connected_clients:
        return

    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)

    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

@app.get("/api/status")
async def get_scanner_status():
    """Get current scanner status"""
    if not scanner_manager:
        return {"scanner_running": False}

    return {
        "scanner_running": scanner_manager.is_scanning,
        "mode": getattr(scanner_manager, 'current_mode', 'flash'),
        "timeframe": getattr(scanner_manager, 'current_timeframe', '60'),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/signals")
async def get_signals(timeframe: str = "1m", mode: str = "flash", asset: str = "ALL"):
    """
    Get current signals filtered by timeframe, mode, and asset
    
    Query params:
    - timeframe: 30s, 1m, 2m, 3m, 5m, 10m, 15m (default: 1m)
    - mode: flash or super (default: flash)
    - asset: specific asset or ALL (default: ALL)
    """
    if not scanner_manager:
        logger.warning("Scanner manager not initialized")
        return []
    
    # Convert timeframe from frontend format (1m) to backend format (60)
    timeframe_map = {
        "30s": "30",
        "1m": "60",
        "2m": "120",
        "3m": "180",
        "5m": "300",
        "10m": "600",
        "15m": "900"
    }
    
    timeframe_seconds = timeframe_map.get(timeframe, timeframe)
    
    logger.info(f"📡 /api/signals called: timeframe={timeframe} ({timeframe_seconds}s), mode={mode}, asset={asset}")
    
    # Get all signals
    all_signals = scanner_manager.get_live_signals() or []
    
    # Filter by timeframe and mode
    filtered_signals = [
        s for s in all_signals 
        if str(s.get('timeframe')) == str(timeframe_seconds) and 
           s.get('mode', 'flash').lower() == mode.lower()
    ]
    
    # Filter by asset if not ALL
    if asset != "ALL":
        filtered_signals = [s for s in filtered_signals if s.get('asset') == asset]
    
    logger.info(f"📤 Returning {len(filtered_signals)} signals (from {len(all_signals)} total)")
    
    return filtered_signals

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
