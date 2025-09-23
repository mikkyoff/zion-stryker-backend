"""
Firebase Service for Zion Stryker Trading Bot
Handles push notifications to mobile devices
"""
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    messaging = None
import logging
import os
from typing import Dict, Any
import json

from .models import SignalData

logger = logging.getLogger(__name__)

class FirebaseService:
    """Firebase service for push notifications"""
    
    def __init__(self):
        self.app = None
        self.initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # For development, we'll create a simple mock service
            # In production, you would initialize with proper credentials
            
            firebase_config = {
                "apiKey": os.getenv("FIREBASE_API_KEY"),
                "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
                "projectId": os.getenv("FIREBASE_PROJECT_ID"),
                "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
                "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
                "appId": os.getenv("FIREBASE_APP_ID"),
                "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
            }
            
            # Check if Firebase is available and configured
            if not FIREBASE_AVAILABLE or not firebase_config.get("projectId"):
                logger.warning("🟡 Firebase not available - notifications will be mocked")
                self.initialized = False
                return
            
            # Initialize Firebase (in production, use service account key)
            if not firebase_admin._apps:
                # For development/demo purposes, we'll mock Firebase
                logger.info("🟡 Firebase in mock mode for development")
                self.initialized = True
            else:
                self.app = firebase_admin.get_app()
                self.initialized = True
                
            logger.info("✅ Firebase service initialized")
            
        except Exception as e:
            logger.warning(f"🟡 Firebase initialization failed: {e}")
            self.initialized = False
    
    async def send_signal_notification(self, signal: SignalData) -> bool:
        """Send push notification for new trading signal"""
        try:
            if not self.initialized:
                # Mock notification for development
                await self._mock_notification(signal)
                return True
            
            # Create notification message
            notification_title = f"Zion Stryker - {signal.priority.upper()} Signal"
            
            if signal.mode == "flash":
                notification_body = f"{signal.asset} {signal.direction} {signal.priority}"
            else:  # super mode
                notification_body = f"{signal.asset} {signal.direction}"
            
            # Prepare message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification_title,
                    body=notification_body
                ),
                data={
                    "signal_id": signal.signal_id,
                    "asset": signal.asset,
                    "direction": signal.direction,
                    "priority": signal.priority,
                    "mode": signal.mode,
                    "timeframe": signal.timeframe,
                    "timestamp": signal.timestamp.isoformat(),
                    "confidence": str(signal.confidence)
                },
                # In production, you would target specific topics or tokens
                topic="trading_signals"
            )
            
            # Send notification
            response = messaging.send(message)
            logger.info(f"📱 Push notification sent: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send push notification: {e}")
            return False
    
    async def _mock_notification(self, signal: SignalData):
        """Mock notification for development/testing"""
        notification_data = {
            "title": f"Zion Stryker - {signal.priority.upper()} Signal",
            "body": f"{signal.asset} {signal.direction}" + (f" {signal.priority}" if signal.mode == "flash" else ""),
            "data": {
                "signal_id": signal.signal_id,
                "asset": signal.asset,
                "direction": signal.direction,
                "priority": signal.priority,
                "mode": signal.mode,
                "timeframe": signal.timeframe,
                "timestamp": signal.timestamp.isoformat()
            }
        }
        
        logger.info(f"📱 [MOCK] Push notification: {json.dumps(notification_data, indent=2)}")
    
    async def send_scanner_status_notification(self, status: str, mode: str = None, timeframe: str = None) -> bool:
        """Send notification for scanner status changes"""
        try:
            title = "Zion Stryker Scanner"
            
            if status == "started":
                body = f"Scanner started in {mode} mode - {timeframe}s timeframe"
            else:
                body = "Scanner stopped"
            
            if not self.initialized:
                logger.info(f"📱 [MOCK] Scanner notification: {title} - {body}")
                return True
            
            # Send actual notification in production
            message = messaging.Message(
                notification=messaging.Notification(title=title, body=body),
                data={"status": status, "mode": mode or "", "timeframe": timeframe or ""},
                topic="scanner_status"
            )
            
            response = messaging.send(message)
            logger.info(f"📱 Scanner notification sent: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send scanner notification: {e}")
            return False