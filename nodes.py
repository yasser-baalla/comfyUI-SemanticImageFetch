import torch
from comfy.samplers import KSAMPLER
from comfy.k_diffusion.sampling import to_d
from tqdm.auto import trange

def adjust_latent(x, mean_ref, std_ref):
    mean_input = x.mean(dim=(2,3), keepdim=True)
    std_input = x.std(dim=(2,3), keepdim=True) 

    return (x - mean_input) / std_input * std_ref + mean_ref

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

        mean_reference = reference['samples'].mean(dim=(2,3), keepdim=True)
        std_reference = reference['samples'].std(dim=(2,3), keepdim=True)

        output_latent = adjust_latent(input['samples'], mean_reference, std_reference)

        return ({'samples': output_latent}, )

class ColorGradeSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference": ("LATENT", {"tooltip": "The reference image"}),
                "start" : ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "The starting point of the color grading"}),
                "end" : ("INT", {"default": 15, "min": 15, "max": 10000, "tooltip": "The end point of the color grading"}),
                },
            }
    
    RETURN_TYPES = ("SAMPLER",)

    FUNCTION = "create_sampler"

    CATEGORY = "LATENT"
    DESCRIPTION = "sampler to color grade the latent to match the reference latent"
    
    def create_sampler(self, reference, start, end):
        mean_reference = reference['samples'].mean(dim=(0,2,3), keepdim=True)

        std_reference = reference['samples'].std(dim=(2,3), keepdim=True)
        std_reference = std_reference.mean(dim=0, keepdim=True)
        
        @torch.no_grad()
        def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            if end <= start:
                raise ValueError("End must be greater than start.")
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(len(sigmas) - 1, disable=disable):
                if s_churn > 0:
                    gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
                    sigma_hat = sigmas[i] * (gamma + 1)
                else:
                    gamma = 0
                    sigma_hat = sigmas[i]

                if gamma > 0:
                    eps = torch.randn_like(x) * s_noise
                    x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
                denoised = model(x, sigma_hat * s_in, **extra_args)
                d = to_d(x, sigma_hat, denoised)
                if start < i < end:
                    denoised = adjust_latent(denoised, mean_reference, std_reference)
                
                if callback is not None:
                    callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
                
                # Euler method
                x = denoised + d * sigmas[i+1]
            return x
        sampler = KSAMPLER(sample_euler)
        return (sampler, )
        
        



