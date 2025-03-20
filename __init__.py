from .nodes import ColorGradingLatent, SemanticImageFetch, ColorGradeSampler



NODE_CLASS_MAPPINGS = {"ColorGrading": ColorGradingLatent, "SemanticImageFetch": SemanticImageFetch, "ColorGradeSampler": ColorGradeSampler}
NODE_DISPLAY_NAME_MAPPING = {"ColorGrading": "Color Grading", "SemanticImageFetch": "Semantic Image Fetch", "ColorGradeSampler": "Color Grade Sampler"}