"""
Firebase Service for push notifications
"""
import logging
import os
from typing import Optional
from .models import SignalData

logger = logging.getLogger(__name__)

class FirebaseService:
    """Firebase service for push notifications"""
    
    def __init__(self):
        self.initialized = False
        try:
            import firebase_admin
            from firebase_admin import credentials, messaging
            
            # Try to initialize Firebase
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.initialized = True
                self.messaging = messaging
                logger.info("✅ Firebase initialized successfully")
            else:
                logger.warning("⚠️  Firebase credentials not found - notifications disabled")
        except Exception as e:
            logger.warning(f"⚠️  Firebase initialization failed: {e}")
    
    async def send_signal_notification(self, signal: SignalData):
        """Send push notification for new signal"""
        if not self.initialized:
            return
        
        try:
            message = self.messaging.Message(
                notification=self.messaging.Notification(
                    title=f"🎯 {signal.priority.upper()} Signal - {signal.asset}",
                    body=f"{signal.direction.upper()} | {signal.mode} mode | {signal.confidence:.1f}% confidence"
                ),
                topic="trading_signals"
            )
            
            response = self.messaging.send(message)
            logger.info(f"📱 Notification sent: {response}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
