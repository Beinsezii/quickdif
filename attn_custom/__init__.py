from .subquad import SubQuadraticCrossAttnProcessor

try:
    from .rocm_flash import FlashAttnProcessor
except ImportError:
    pass
