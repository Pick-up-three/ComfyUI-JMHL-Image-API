from .nodes import PixelAPIImage2Node

NODE_CLASS_MAPPINGS = {
    "PixelAPIImage2Node": PixelAPIImage2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelAPIImage2Node": "JMHL Image API",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
