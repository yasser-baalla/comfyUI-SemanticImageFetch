from .nodes import ColorGradingLatent, SemanticImageFetch



NODE_CLASS_MAPPINGS = {"ColorGrading": ColorGradingLatent, "SemanticImageFetch": SemanticImageFetch}
NODE_DISPLAY_NAME_MAPPING = {"ColorGrading": "Color Grading", "SemanticImageFetch": "Semantic Image Fetch" }