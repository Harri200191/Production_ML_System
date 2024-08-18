import os
from pymongo import MongoClient 
import re
from app import retrieve_image_from_image, crop_image_using_yolo, fetch_coordinates_and_draw_box_and_label
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
from db_connect import start_connection 
import json
import numpy as np
from app import crop_image_using_yolo
from clip_embeddings import ClipEmbeddingsfunction
import joblib
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

clip_embedder = ClipEmbeddingsfunction(device="cuda")
collection, MONGO_URI = start_connection()

class EvaluateModel():
    def __init__(self, device): 
        self.device = device
        self.client = MongoClient(MONGO_URI) 

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

    def plot_confusion_matrix(self, list1_labels_predicted, list2_labels_actual):
        """
            This function takes the predicted labels and actual labels and plots the confusion matrix.

            Inputs:
                list1_labels_predicted (list): The list of predicted labels.
                list2_labels_actual (list): The list of actual labels.

            Outputs:
                None
        """

        confusion_counts = defaultdict(int) 
        for predicted_label, actual_label in zip(list1_labels_predicted, list2_labels_actual):
            if predicted_label == actual_label:
                confusion_counts[(predicted_label, actual_label)] += 1
 
            labels = sorted(set(list1_labels_predicted + list2_labels_actual)) 
            conf_matrix = confusion_matrix(list2_labels_actual, list1_labels_predicted, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        # Save the confusion matrix in the results directory
        if not os.path.exists("results"):
            os.makedirs("results")
        
        plt.savefig("`results/`confusion_matrix.png") 
        print("Saved confusion matrix in results directory")

    def evaluate(self, test_directory):
        """
            This function evaluates the model by comparing the predicted labels with the actual labels.

            Inputs:
                None

            Outputs:
                list: The list of predicted labels.
                list: The list of actual labels.
        """
        
        image_filenames = os.listdir(test_directory)
        image_filenames = [filename.split(".")[0] for filename in image_filenames]

        try:
            db = self.client['veeve-eip-uae']
        except:
            print("Error: Could not connect to MongoDB")
            return "" 
        
        collection = db['frames'] 
        query = {"frameName": {"$in": image_filenames}}
        cursor = collection.find(query)
        frame_names_not_to_add = []

        print("Evalutaing model.....")
        labels_predicted =  []
        for x in image_filenames:
            temp_list = []
            output = retrieve_image_from_image(f"./{test_directory}/{x}.png")
            if output == None:
                frame_names_not_to_add.append(x)
                continue

            match = re.search(r'label: (\w+)', output[0][1])
            label = match.group(1) 
            temp_list.append(label)
            temp_list.append(x)
            labels_predicted.append(temp_list) 
  
        labels_actual = []
        for document in cursor:   
            temp_list = []
            frame_name = document["frameName"]
            
            if (frame_name in frame_names_not_to_add):
                continue

            try:
                label = document["annotations"][0]["boxes"][0]["label"]  
            except:
                label = document["annotations"][1]["boxes"][0]["label"]  
            
            temp_list.append(label)
            temp_list.append(frame_name)

            labels_actual.append(temp_list)

        dict1 = {frame_name: label for label, frame_name in labels_predicted}
        dict2 = {frame_name: label for label, frame_name in labels_actual}
 
        list1_labels_predicted = [dict1[frame_name] for _, frame_name in labels_actual if frame_name in dict1]
        list2_labels_actual = [dict2[frame_name] for _, frame_name in labels_actual if frame_name in dict2]
 
        return list1_labels_predicted, list2_labels_actual
    
    def evaluate_for_single_image_using_yolo(self, input_image):
        """
            This function takes an image as input and returns the predicted label.

            Inputs:
                input_image (str): The path to the image file.
                
            Outputs:
                str: The predicted label. 
        """ 

        print("Evalutaing image using yolo.....") 
        cropped_image, flag, label_yolo = crop_image_using_yolo(input_image) 
        if flag == -1:
            return "No label found"
        else:
            return str(label_yolo) 
    
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
    
    def evaluate_for_video(self, video_path):  
        """
            This function evaluates the model by comparing the predicted labels with the actual labels for a video.

            Inputs:
                video_path (str): The path to the video file.

            Outputs:
                int: The number of frames in the video.
        """
        cap = cv2.VideoCapture(video_path) 


        if not cap.isOpened():
            print("Error: Unable to open video file")
            exit() 


        video_name = video_path.split(".")[0]
        count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(f'{video_name}_results.mp4', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 

        flag = False     
        counter = 0
        last_prediction = None

        with open("barcode_mappings.json", "r") as f:
            barcode_data = json.load(f)

        while True: 
            ret, frame = cap.read() 
            if not ret:
                break 
 
            inference_result = retrieve_image_from_image(frame) 
            if inference_result == None and flag == False:
                out.write(frame)
                continue
            else:
                if inference_result != None:
                    y0, dy = 50, 40

                    for i, (_, desc, distance) in enumerate(inference_result): 
                        flag = True 
                        label = None
                        barcode = None

                        lines = desc.split('\n') 

                        distances = distance
                        
                        for line in lines:
                            if line.startswith('label:'):
                                label = line.split(': ')[1].strip('-')   
                            elif line.startswith('barcode:'):
                                barcode = line.split(': ')[1].strip('-')   
                                
                        product_name = None
                        for record in barcode_data:
                            if record["barcode"] == barcode.strip():
                                product_name = record["name"]
                            
                        if product_name == None:
                            product_name = "Unknown"

                        final_str = "Label: {} Barcode: {} Product Name: {} Distance: {:.4f}".format(label, barcode, product_name, distances)
                        last_prediction = final_str 

                        counter = 0   

                        if last_prediction is not None:
                            y = y0 + i * dy
                            text = str(last_prediction)
                            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA) 

                        count += 1
                        counter += 1

                        if count % 100 == 0:
                            os.system("rm -r runs")

                        if counter >= 30:
                            flag = False
                            last_prediction = None  
            
                    out.write(frame) 
                else:
                    out.write(frame) 
                    continue

        cap.release()
        cv2.destroyAllWindows()
        return count

    def evaluate_for_video_using_svm(self, video_path, save_as):  
        """
            This function evaluates the model by comparing the predicted labels with the actual labels for a video.

            Inputs:
                video_path (str): The path to the video file.

            Outputs:
                int: The number of frames in the video.
        """ 

        cap = cv2.VideoCapture(video_path) 

        if not cap.isOpened():
            print("Error: Unable to open video file")
            exit() 

        with open("barcode_mappings.json", "r") as f:
            meta_data = json.load(f)

        count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(f'{save_as}.mp4', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 
    
        while True: 
            ret, frame = cap.read() 
            if not ret:
                break 
            
            cropped_image, status, label_yolo = crop_image_using_yolo(frame)
            if status == -1:
                out.write(frame) 
                continue
                
            new_embedding = clip_embedder.embed_image(cropped_image) 
            svm_model = joblib.load("svm_model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            barcode = self.predict_barcode_using_svm(new_embedding, svm_model, label_encoder) 

            label = None 
            product_name = None

            for record in meta_data:
                if record["barcode"] == barcode:
                    product_name = record["name"]
                    label = record["shape_category"]

            if label!=label_yolo: 
                out.write(frame) 
                continue
                    
            final_str = "Label: {} AND Barcode: {} Product Name: {}".format(label, barcode, product_name)
            last_prediction = final_str 

            if last_prediction is not None:
                text = "Result: " + str(last_prediction)
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
            
        cap.release()
        cv2.destroyAllWindows()
        return count
    
    def make_predictions_for_yolo_only(self, input_video):
        """
            This function makes predictions using YOLO only.

            Inputs:
                input_video (str): The path to the video file.

            Outputs:
                None
        """

        cap = cv2.VideoCapture(input_video) 
        video_name = input_video.split(".")[0]

        if not cap.isOpened():
            print("Error: Unable to open video file")
            exit() 

        count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(f'{video_name}_yolo_results.mp4', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) )

        while True: 
            ret, frame = cap.read() 
            if not ret:
                break 

            img = fetch_coordinates_and_draw_box_and_label(frame)  
            out.write(img)
            
        cap.release()
        cv2.destroyAllWindows()
        return count
        