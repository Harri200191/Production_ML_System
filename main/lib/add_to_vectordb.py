import os
from chromadb import Client, Settings

from lib.clip_embeddings import ClipEmbeddingsfunction
from lib.config import Config 
from lib.data_extraction import DataExtraction

class DataEmbeddings:
    def __init__(self, modelName):
        self.configurations = Config()
        self.data_extraction = DataExtraction()
        self.embedding_function = ClipEmbeddingsfunction(model_name=modelName)
        self.client = Client(settings = Settings(is_persistent=True, persist_directory="./clip_chroma"))
        self.collection = self.create_collection()

    def create_collection(self):
        """
            Create a collection in ChromaDB.
        """

        collection = self.client.get_or_create_collection(
            name = self.configurations.MODEL_CONFIGS["CHROMA_VERSION"], 
            embedding_function = self.embedding_function,
            metadata= {f"hnsw:space": self.configurations.MODEL_CONFIGS["SIMILARITY_ALGO"]}
        )

        return collection 

    def add_embeddings_to_chroma(self):
        """
            Add image embeddings and metadata to the ChromaDB collection.

            This function retrieves image file paths and corresponding metadata from a JSON file,
            adds the images and metadata to a ChromaDB collection, and assigns unique IDs to each image.
        """

        data_dict = {}
        list_of_dicts = []

        try:
            path_to_images, labels = self.data_extraction.read_data(self.configurations.DATASET_CONFIGS["IMAGES_DIR"])
        except FileNotFoundError:
            print("Unexpected!")
            return  
        
        print("Filled the data. Now adding to chroma........ ") 

        # create metadata dictionary having records of image path and label
        for i in range(len(path_to_images)):
            data_dict[str(i)] = {"path": path_to_images[i], "label": labels[i]}
            list_of_dicts.append(data_dict)
    
        self.collection.add(
            ids=[str(i) for i in range(len(path_to_images))],
            documents = path_to_images,
            metadatas = list_of_dicts,
        )
        
        print("Added to chroma!") 