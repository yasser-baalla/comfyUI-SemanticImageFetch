import torch


class SemanticImageFetch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The list of images to fetch the semantic map from."}), 
                "conditionning": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "number_of_candidates": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Number of closest images to retrieve."}),
            }
        }
    RETURN_TYPES = ("IMAGE", "CONDITIONING",)

    FUNCTION = "fetch"

    CATEGORY = "conditioning"
    DESCRIPTION = "Select the closest images to an input prompt"
    
    def fetch(self, image, conditioning, prompt, number_of_candidates):
    # Encode the prompt using the CLIP model
        with torch.no_grad():
            text_embeddings = conditioning.encode(prompt)

        # Encode the images using the CLIP model
        image_embeddings = self.encode_images(image, conditioning)

        # Compute similarity between text and image embeddings
        similarities = self.compute_similarity(text_embeddings, image_embeddings)

        # Select the top-k closest images
        top_k_indices = torch.topk(similarities, k=number_of_candidates, dim=0).indices
        closest_images = image[top_k_indices]

        return (closest_images, conditioning)