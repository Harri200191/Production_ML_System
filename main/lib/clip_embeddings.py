import clip
import torch
from numpy import ndarray
from typing import List
from PIL import Image 
from chromadb import EmbeddingFunction, Documents, Embeddings

from lib.config import Config
Configurations = Config()

class ClipEmbeddingsfunction(EmbeddingFunction):
    def __init__(
            self, 
            model_name: str = f"{Configurations.MODEL_CONFIGS["CLIP_MODEL_NAME"]}", 
            device: str = Configurations.MODEL_CONFIGS["DEVICE"]
        ):  

        """
            Initialize the ClipEmbeddingsfunction.

            Args:
                model_name (str, optional): The name of the CLIP model to use. Defaults to "ViT-B/32".
                device (str, optional): The device to use for inference (e.g., "cpu" or "cuda"). Defaults to "cpu".
        """

        self.device = device
        self.model, self.preprocess = clip.load(model_name, self.device)

    def __call__(self, input: Documents)-> Embeddings:
        """
            Compute embeddings for a batch of images.

            Args:
                input (Documents): A list of image file paths.

            Returns:
                Embeddings: A list of image embeddings.
        """
            
        list_of_embeddings = []

        for image_path in input: 
            image = Image.open(image_path)
            image = image.resize((64, 64))
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_image(image_input).cpu().detach().numpy()

            list_of_embeddings.append([float(value) for value in embeddings[0]]) 

        return list_of_embeddings
    
    def embed_image(self, input: ndarray)-> ndarray:  
        """
            Compute embeddings for a single image.

            Args:
                input (str): The file path of the image or "cropped_image.png".

            Returns:
                ndarray: The image embedding.
        """

        # Convert ndarray to PIL Image to handle resizing
        input_image = Image.fromarray(input)
        input_image = input_image.resize((64, 64), Image.ANTIALIAS)

        # Preprocess the image and prepare for model input
        input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)

        # Compute embeddings with no gradient calculation
        with torch.no_grad():
            embeddings = self.model.encode_image(input_tensor).cpu().detach().numpy()

        return embeddings[0]
     
    