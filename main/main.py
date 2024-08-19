from lib.config import Config
from lib.add_to_vectordb import DataEmbeddings

import json
import argparse

def load_model_choices():
    with open(Config.MODEL_CONFIGS['MODEL_CHOICES_JSON'], 'r') as f:
        data = json.load(f) 
        model_choices = data['models']

    return model_choices 

def parse_args(): 
    model_choices = load_model_choices() 

    parser = argparse.ArgumentParser(description="IMAGE NET SEARCH TOOL") 

    parser.add_argument('--model-name', type=str, choices=model_choices,
                        required=True, help='Models to train')  
    parser.add_argument('--add_to_db', type=int, default=0,
                        help='Start Training or Not') 
    parser.add_argument('--device', type=int, nargs='+', required=False,
                        default=[0], help='Device(s) to use for training (e.g., [0], [0,1])')

    return parser.parse_args()
 
if __name__ == "__main__":
    arguments = parse_args()
    print(arguments)

    data_embed = DataEmbeddings(arguments.model_name)

    if arguments.add_to_db: 
        data_embed.add_embeddings_to_chroma()
 
    
