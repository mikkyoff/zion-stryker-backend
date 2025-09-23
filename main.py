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

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zion-stryker-frontend.vercel.app",
        "https://zion-stryker-frontend2.vercel.app", 
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "https://*.vercel.app"  # Any Vercel deployment
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global state
scanner_manager: Optional[ScannerManager] = None
connected_clients: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global scanner_manager
    logger.info("Starting Zion Stryker Backend...")
    
    try:
        # Initialize Firebase service
        firebase_service = FirebaseService()
        
        # Initialize signal engine
        signal_engine = SignalEngine()
        
        # Initialize scanner manager
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

@app.get("/api/signals")
async def get_signals(timeframe: str = "1m", mode: str = "Flash", asset: str = "ALL"):
    """Get signals with filters - required by frontend"""
    if not scanner_manager:
        return []
    
    try:
        live_signals = scanner_manager.get_live_signals()
        
        # Apply frontend filters
        filtered_signals = []
        for signal in live_signals:
            # Filter by timeframe
            if timeframe != "ALL":
                signal_timeframe = signal.get("timeframe", "")
                # Convert timeframe formats (e.g., "60" -> "1m", "120" -> "2m")
                if timeframe == "1m" and signal_timeframe != "60":
                    continue
                elif timeframe == "2m" and signal_timeframe != "120":
                    continue
                elif timeframe == "3m" and signal_timeframe != "180":
                    continue
                elif timeframe == "5m" and signal_timeframe != "300":
                    continue
                elif timeframe == "10m" and signal_timeframe != "600":
                    continue
                elif timeframe == "15m" and signal_timeframe != "900":
                    continue
                elif timeframe == "30s" and signal_timeframe != "30":
                    continue
            
            # Filter by mode
            if mode != "ALL":
                signal_mode = signal.get("mode", "").lower()
                if signal_mode != mode.lower():
                    continue
            
            # Filter by asset
            if asset != "ALL" and signal.get("asset") != asset:
                continue
            
            # Transform signal for frontend compatibility
            frontend_signal = {
                "asset": signal.get("asset", ""),
                "direction": signal.get("direction", ""),
                "timeframe": signal.get("timeframe", ""),
                "confidence": signal.get("confidence", 0),
                "urgency": signal.get("priority", "normal"),  # Map priority to urgency
                "reasons": [signal.get("priority", "")],  # Convert priority to reasons array
                "expiry": int(signal.get("timeframe", "60"))  # Use timeframe as expiry
            }
            
            filtered_signals.append(frontend_signal)
        
        return filtered_signals
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []

@app.get("/api/status")
async def get_scanner_status():
    """Get scanner status - required by frontend"""
    if not scanner_manager:
        return {"scanner_running": False}
    
    return {
        "scanner_running": scanner_manager.is_scanning,
        "mode": getattr(scanner_manager, 'current_mode', None),
        "timeframe": getattr(scanner_manager, 'current_timeframe', None),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/scanner/toggle")
async def toggle_scanner(request: dict):
    """Toggle scanner - handle both frontend and backend formats"""
    global scanner_manager
    
    if not scanner_manager:
        raise HTTPException(status_code=500, detail="Scanner manager not initialized")
    
    try:
        # Handle frontend format: {"enabled": true/false}
        if "enabled" in request:
            action = "start" if request["enabled"] else "stop"
            mode = "flash"  # Default mode
            timeframe = "60"  # Default timeframe (1 minute)
        # Handle backend format: {"action": "start"|"stop", "mode": "flash", "timeframe": "60"}
        else:
            action = request.get("action", "stop")
            mode = request.get("mode", "flash")
            timeframe = request.get("timeframe", "60")
        
        if action == "start":
            result = await scanner_manager.start_scanner(mode=mode, timeframe=timeframe)
            
            # Broadcast to all connected clients
            await broadcast_to_clients({
                "type": "scanner_status",
                "status": "started",
                "mode": mode,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            })
            
        else:  # stop
            result = await scanner_manager.stop_scanner()
            
            await broadcast_to_clients({
                "type": "scanner_status", 
                "status": "stopped",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "action": action,
            "message": result,
            "enabled": action == "start",  # Add this for frontend compatibility
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error toggling scanner: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to Zion Stryker signals",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle incoming messages if needed
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
    """Broadcast message to all connected WebSocket clients"""
    if not connected_clients:
        return
    
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))  # Default to 5000, Railway will override
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )