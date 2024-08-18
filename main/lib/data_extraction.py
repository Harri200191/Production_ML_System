import os
import pandas as pd
import json
from tqdm import tqdm

from lib.config import Config

class DataExtraction:
    # Default constructor
    def __init__(self):
        self.configurations = Config()  

    # create labels number to value mapping dictionary and store as json
    def create_labels_dict(self, file_path):
        """
            Create a dictionary mapping the class number to the class name and store it as a JSON file.

            Args:
                file_path (str): The path to the text file containing the class number to class name mapping.

            Returns:
                dict: A dictionary mapping the class number to the class name.
        """

        mapping = {}
        # Check if the file exists
        if not os.path.exists(file_path):
            raise Exception("File not found")
        
        data = pd.read_csv(file_path, sep="\t", header=0, names=["ID", "Class"])
        labels = data["Class"].unique()
        
        # Read the data from the text file
        data = pd.read_csv(file_path, sep=" ", header=None)

        # Create a dictionary to store the mapping
        for value in data.values:
            key = value[0]
            label = value[1]
            mapping[key] = label 

        with open(self.configurations.DATASET_CONFIGS["LABELS_DICT_JSON_PATH"], "w") as f:
            json.dump(mapping, f)

        return mapping

    # Function to read the data from the file
    def read_data(self, folder_path):
        """
            Read the data from the file.

            Args:
                folder_path (str): The path to the directory containing the images.

            Returns:
                list: A list of image file paths.
                list: A list of labels
        """
        
        # Check if the file exists
        if not os.path.exists(folder_path):
            raise Exception("Directory not found")
        
        mappings = self.create_labels_dict(self.configurations.DATASET_CONFIGS["LABELS_TXT_PATH"])
        path_to_image = []
        labels = []
         
        for file in tqdm(os.listdir(folder_path)):
            complete_path = os.path.join(folder_path, file)
 
            for image in os.listdir(complete_path):
                path_to_image.append(os.path.join(complete_path, image))
                
            labels.append(mappings[file])

        return path_to_image, labels


            