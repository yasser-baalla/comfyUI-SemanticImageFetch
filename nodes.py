import torch

class SemanticImageFetch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The list of images to fetch the semantic map from."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "prompt": ("STRING",{"multiline": True} ),
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
        image_embed = clip_vision.encode_image(image)['image_embeds']
        
        clip_embed = clip_embed / clip_embed.norm(dim=-1, keepdim=True)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarities = torch.matmul(clip_embed, image_embed.T).reshape(-1)

        # Select the top-k closest images
        top_k_indices = torch.topk(similarities, k=number_of_candidates, dim=0).indices.squeeze(0)
        return (image[top_k_indices], )
    
class ColorGradingLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LATENT", {"tooltip": "The latent to be color graded"}),
                "reference": ("LATENT", {"tooltip": "The reference latent"}),
                },
            }
    
    RETURN_TYPES = ("LATENT",)

    FUNCTION = "grade"

    CATEGORY = "LATENT"
    DESCRIPTION = "Take a latent and color grade it to match the reference latent"
    
    def grade(self, input, reference):
        mean_input = input['samples'].mean(dim=(0,2,3), keepdim=True)
        mean_reference = reference['samples'].mean(dim=(0,2,3), keepdim=True)
        std_input = input['samples'].std(dim=(0,2,3), keepdim=True)
        std_reference = reference['samples'].std(dim=(0,2,3), keepdim=True)

        output_latent = (input['samples'] - mean_input) / std_input * std_reference + mean_reference

        return ({'samples': output_latent}, )