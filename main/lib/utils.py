import os
import re
from app import retrieve_image_from_image, crop_image_using_yolo, fetch_coordinates_and_draw_box_and_label
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2  
import json
import numpy as np 
from clip_embeddings import ClipEmbeddingsfunction
import joblib
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

clip_embedder = ClipEmbeddingsfunction(device="cuda") 
class EvaluateModel():
    def __init__(self, device): 
        self.device = device 

    def load_data_and_train_svm(self): 
        """
            This function loads the data from the MongoDB database and trains an SVM model.

            Inputs:
                None

            Outputs:
                None 
        """
        
        try:
            db = self.client['veeve-eip-uae']
        except:
            print("Error: Could not connect to MongoDB") 

        collection = db['frames']      
        image_paths = glob.glob('cropped_imgs/*.png') 

        image_filenames = [filename.split("/")[1].split(".")[0] for filename in image_paths]  
        query = {"frameName": {"$in": image_filenames}}
        documents = collection.find(query)

        img_details_list = [] 
        count = 0
        total = 0

        for document in documents:  
            img_dict = {}   
            total += 1
            try:
                label = document["annotations"][0]["boxes"][0]["label"] 
                barcode = document["annotations"][0]["boxes"][0]["barcode"]  
                frame_name = document["frameName"]
                frame_name = "cropped_imgs/"+frame_name+".png"
                
                if frame_name == None or label == None:
                    label = document["annotations"][1]["boxes"][0]["label"] 
                    barcode = document["annotations"][1]["boxes"][0]["barcode"]  
                    frame_name = document["frameName"]
                    frame_name = "cropped_imgs/"+frame_name+".png"

                    if frame_name == None or label == None:
                        continue 

                img_dict["barcode"] = barcode 
                img_dict["label"] = label
                img_dict["frame_name"] = frame_name              
                img_details_list.append(img_dict)    
            except:
                print("Nothing found")
                count += 1
                continue 
        
        print("Total amount of images: ", total)
        print("Amount of images not found: ", count)
        meta_results = []
        for details in img_details_list:
            image = details["frame_name"]
            img = cv2.imread(image)
            results = clip_embedder.embed_image(img) 
            meta_results.append(results)
        
        temp = np.array(meta_results)
        data = img_details_list 
        embeddings = temp  
        barcodes = [item['barcode'] for item in data]
        embeddings = np.array(embeddings) 
        label_encoder = LabelEncoder()
        encoded_barcodes = label_encoder.fit_transform(barcodes) 
        joblib.dump(label_encoder, "label_encoder.pkl")
        svm_model = svm.SVC(kernel='linear', probability=True)
        svm_model.fit(embeddings, encoded_barcodes) 
        joblib.dump(svm_model, "svm_model.pkl")

    def predict_barcode_using_svm(self, embedding, model, label_encoder):
        """
            This function takes an image embedding, SVM model, and label encoder, and returns the predicted barcode.

            Inputs:
                embedding (list): The image embedding.
                model (SVC): The SVM model.
                label_encoder (LabelEncoder): The label encoder.

            Outputs:
                str: The predicted barcode.
        """

        embedding = np.array(embedding).reshape(1, -1)
        encoded_label = model.predict(embedding)
        barcode = label_encoder.inverse_transform(encoded_label)
        return barcode[0]
    
    def evaluate_for_single_image(self, input_image):
        """
            This function takes an image as input and returns the predicted label.

            Inputs:
                input_image (str): The path to the image file.
                
            Outputs:
                str: The predicted label. 
        """ 

        print("Evalutaing image.....") 
        output = retrieve_image_from_image(input_image) 

        if output == None:
            return "No label found"
        else:
            match = re.search(r'label: (\w+)', output[0][1])
            label = match.group(1)  
   
        return str(label)