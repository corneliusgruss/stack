"""Data collection, processing, and loading utilities."""

from stack.data.iphone_loader import iPhoneSession, load_session, verify_session

# Encoder and dataset imports are optional (need pyserial / hardware)
try:
    from stack.data.encoder import EncoderReceiver, EncoderReading
except ImportError:
    pass

try:
    from stack.data.dataset import DemoDataset
except ImportError:
    pass

__all__ = [
    "iPhoneSession",
    "load_session",
    "verify_session",
]
