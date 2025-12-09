import base64
from pathlib import Path
from typing import Union
from PIL import Image
import io


def encode_image_base64(image_source: Union[str, bytes, Image.Image]) -> str:
    """
    Encode an image to base64 for the multimodal API.
    
    Args:
        image_source: Path to image file, raw bytes, or PIL Image
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image_source, str):
        # File path
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_source, Image.Image):
        # PIL Image
        buffer = io.BytesIO()
        image_source.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    else:
        # Already bytes
        image_bytes = image_source
    
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")
