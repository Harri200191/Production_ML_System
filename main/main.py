from lib.config import Config
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
    parser.add_argument('--organization', type=int,
                        required=True, help='Organization ID')
    parser.add_argument('--store', type=int, nargs='+',
                        required=True, help='Store id')
    parser.add_argument('--model-name', type=str, choices=model_choices,
                        required=False, default=model_choices[0], help='Models to train')
    parser.add_argument('--model-type', type=str,
                        choices=['aligned', 'oriented'], required=False, help='Model Type to train', default='oriented')
    parser.add_argument('--dataset-version', type=str, required=False,
                        help='Dataset version (latest or int value)', default=-1)
    parser.add_argument('--extract-data', type=bool, required=False, default=False,
                        help='extract the new data or not')
    parser.add_argument('--train', type=bool, required=False, default=False,
                        help='Start Training or Not')
    parser.add_argument('--epochs', type=int, required=False,
                        default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, required=False,
                        default=32, help='Batch size for training')
    parser.add_argument('--device', type=int, nargs='+', required=False,
                        default=[0], help='Device(s) to use for training (e.g., [0], [0,1])')

    return parser.parse_args()

 
if __name__ == "__main__":
    arguments = parse_args()
 
    
