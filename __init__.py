class SemanticImageFetch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The list of images to fetch the semantic map from."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "text": ("STRING", ),
                "clip_vision": ("CLIP_VISION", {"tooltip": "The CLIPVision model used for encoding the images."}),
                "number_of_candidates": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Number of closest images to retrieve."}),
            }
        }
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "fetch"

    CATEGORY = "IMAGE"
    DESCRIPTION = "Select the closest images to an input prompt"
    
    def fetch(self, image, clip, clip_vision, prompt, number_of_candidates):
        # Encode the prompt using the CLIP model
        tokens = clip.tokenize(prompt)
        clip_embed = clip.encode_from_tokens_scheduled(tokens)[0][1]['pooled_output']
        
        # Encode the images using the CLIP model
        # check if the projected image is the one we want
        image_embeddings = clip_vision.encode_image(image)['image_embeds']

        print(image_embeddings.shape, clip_embed.shape)
        # # Compute similarity between text and image embeddings
        # similarities = self.compute_similarity(clip_embed, image_embeddings)

        # # Select the top-k closest images
        # top_k_indices = torch.topk(similarities, k=number_of_candidates, dim=0).indices
        # closest_images = image[top_k_indices]

        return (image[:number_of_candidates], )
    

NODE_CLASS_MAPPINGS = {"SemanticImageFetch": SemanticImageFetch}
NODE_DISPLAY_NAME_MAPPING = {"SemanticImageFetch": "Semantic Image Fetch"}