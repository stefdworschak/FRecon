import os
import cv2 as cv
import numpy  as np
import ntpath
import math
import pickle
from scipy.special import softmax


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class FRecon :
    #FRecon class is built to be used as part os facial recognition presentation
    #All the life cycle of a simple facial recognition feature is covered by this class.
    #It includes detect faces, extract face keypoints, training ML model SVM or LR and make predictions

    
    def __init__(self, model_name,relative_path):

        #the constructor will load all DNN models that are required to detect a face in the image
        #then extract keypoints of that face and apply the classifier
        #Also it loads ML trained model and the class labels


        #CAFE model load
        self.relative_path = relative_path
        self.model_dnn_type = model_name
        self.detecta_rosto = cv.dnn.readNetFromCaffe(self.relative_path+'modelDNN/deploy.prototxt', self.relative_path+'modelDNN/res10_300x300_ssd_iter_140000.caffemodel')
        self.detecta_rosto.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.detecta_rosto.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


        #OPENFACE load model
        self.open_face = cv.dnn.readNetFromTorch(self.relative_path+"modelDNN/openface/openface.nn4.small2.v1.t7");
        self.open_face.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.open_face.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        #FACENET load model
        self.face_net = cv.dnn.readNetFromTensorflow(self.relative_path+"modelDNN/facenet/facenet_graph_final.pb");
        self.face_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


        #ML SVM model load
        model_name = self.relative_path+"modelSVM/SVC_model.pkl"
        if os.path.exists(model_name):
            with open(model_name, 'rb') as file:  
                self.svc = pickle.load(file)

        #ML LR model load        
        path_name = self.relative_path+"modelLR/LR_model.pkl"
        if os.path.exists(path_name):
            with open(path_name, 'rb') as file:  
                self.logistic_r = pickle.load(file)
        


        path_model = self.relative_path+"modelSVM/etiquetas.pkl"
        if os.path.exists(path_model):
            with open(path_model, 'rb') as file:  
                self.image_labels_svm = pickle.load(file)
   
        path_model = self.relative_path+"modelLR/etiquetas.pkl"
        if os.path.exists(path_model):
            with open(path_model, 'rb') as file:  
                self.image_labels_lr = pickle.load(file)
            
     
    def AbsoluteFilePaths(self, directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
                
    def ExtractDatasetData(self, path):
        #this method checks for images in path parameters and extract info from file's names like classID and file name to be loaded
        #for example the file name would be 1@ELVIS_PRESLEY-1.jpg. Then the method extracts the classId, it is the number before @
        #also it extracts class label, it is the name between @ and -.
        #the classId is used for training purposes and the class label will be serialized, to be used in future predictions
        image_labels={}
        files=[]
        indice_etiqueta_lista = []
        for file in self.AbsoluteFilePaths(path):
            files.append(file)
            nome_arquivo = ntpath.basename(file)
            index_1 = nome_arquivo.find("@")
            index_2 = nome_arquivo.find("-")

            indice_etiqueta = int(nome_arquivo[:index_1])
            indice_etiqueta_lista.append(indice_etiqueta)
            label_etiqueta = nome_arquivo[index_1+1:index_2]

            if indice_etiqueta not in image_labels :
                image_labels[indice_etiqueta] = label_etiqueta

        return image_labels,files,indice_etiqueta_lista
    
   
    def FindFaceinImage(self, path_image="", img = None):
        #For each one of images found in training images folder, we run thru either "OPENFACE" or "FACENET" DNN models and try to detect face in image
        #When face is found the coordinates of detection is returned to be used later.
        if img is None:
            img = cv.imread(path_image)
            if img is None:
                return None

        blob = cv.dnn.blobFromImage(img, 1.0,(300,300),(104.0, 177.0, 123.0), False, False)
        self.detecta_rosto.setInput(blob)
        detections = self.detecta_rosto.forward()

        return img, detections
        
    
    def ExtractFaceRectangle(self, img, detections, confidence_threshold=0.2):
        #loop over the detections
        #check for threshold confidence before extract boxes of face in image
        boxes = []
        (h, w) = img.shape[:2]
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > confidence_threshold:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, min(startX, w - 1));
                startY = max(0, min(startY, h - 1));
                endX = max(0, min(endX, w - 1));
                endY = max(0, min(endY, h - 1));

                boxes.append([(startX, startY, endX, endY),confidence])

        return boxes
    
    def DrawImageFaceRectangle(self, img, box, confidence, text=None):
        # draw the bounding box of the face along with the associated probability
        # the output of this method contais the original image, with a rectangle drawn to show face detected and the probability of it belongs to any class
        img2 = img.copy()
        if text is None:
            text = "{:.2f}%".format(confidence * 100)
        else :
            text = text + "  {:.2f}%".format(confidence * 100)
        #(startX, startY, endX, endY) = box[0],box[1],box[2],box[3]
        (startX, startY, endX, endY) = box
        y = endY +20 
        cv.rectangle(img2, (startX, startY), (endX, endY),(0, 255, 0), 1)
        cv.putText(img2, text, (startX-20, y),cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return img2;


    def TrainSVC(self, x_train, y_train):
        #method for SVM traing with most standar parameter since there is no enough dataset to apply GridSearch
        #after model is trained it is serialized and saved to folder
        print("Training SVC")
        self.svc = SVC(kernel = "rbf",C=1, gamma=0.1, class_weight="balanced", decision_function_shape = "ovr")
        self.svc.fit(x_train,y_train)

        model_name = self.relative_path+"modelSVM/SVC_model.pkl"  

        with open(model_name, 'wb') as file:  
            pickle.dump(self.svc, file)
        print("Saving SVC model")    
            
    def TrainLogisticRegression(self, x_train, y_train):
        #method for LR traing with most standar parameter since there is no enough dataset to apply GridSearch
        #after model is trained it is serialized and saved to folder
        print("Training Logistic Regression")
        self.logistic_r = LogisticRegression(multi_class='multinomial', solver='lbfgs' ,class_weight="balanced")
        self.logistic_r.fit(x_train,y_train)

        model_name = self.relative_path+"modelLR/LR_model.pkl"  

        with open(model_name, 'wb') as file:  
            pickle.dump(self.logistic_r, file)
        print("Saving LR model")
        
    def ExtractFacePoints(self, img):
        #apply DNN model chosen to extract keypoints of face. It is the data will be used to train and identify each of the classes
        #OPENFACE return 128 keypoints while FACENET return 512 keypoints
        vec = None
        if self.model_dnn_type =="OPENFACE":

            blob_f = cv.dnn.blobFromImage(img, 1.0/255.0,(96,96),(0, 0, 0), True, False)
            self.open_face.setInput(blob_f)
            vec = self.open_face.forward()
            vec = cv.normalize(vec, None,-1, 1, cv.NORM_MINMAX)
        elif self.model_dnn_type =="FACENET":

            blob_f = cv.dnn.blobFromImage(img, 1.0/255.0,(160,160),(0, 0, 0), True, False)
            self.face_net.setInput(blob_f)
            vec = self.face_net.forward()
            vec = cv.normalize(vec, None,-1, 1, cv.NORM_MINMAX)
        return vec
    
    
    def TrainModelfull(self):
        #This method will check for training images folder, extract them and run the algorythim to make ML model available for predictions
        #The main logic is: loop over each image in "images" folder, detect face and extract boxes, apply DNN model over detected boxes
        #After all keypoint were extracted the dataset is built(x_train, y_train) and it is sent for training.


        self.image_labels,files,indice_etiqueta_lista = self.ExtractDatasetData(self.relative_path+"images")

        indexes_ok=[]
        array_keypoints=[]
        for i,f in enumerate(files):
            print("Colecting image : {} -  {}".format(i,f))
            img, detections = self.FindFaceinImage(path_image=f)
            if detections is not None:
                indexes_ok.append(i)
                boxes = self.ExtractFaceRectangle(img,detections,0.5)
                for b in boxes:

                    (startX, startY, endX, endY) = b[0]
                    points = self.ExtractFacePoints(img[startY:endY,startX:endX])
                    if points is not None:
                        array_keypoints.append(points)


        [indice_etiqueta_lista[i] for i in indexes_ok]

        x_train = np.array([arr.flatten() for arr in array_keypoints])
        
        

        
        y_train = np.array(indice_etiqueta_lista)
        y_train = y_train.reshape(y_train.shape[0],1).ravel()

        
        self.TrainSVC(x_train,y_train)
        
            
        self.TrainLogisticRegression(x_train,y_train)
        
        model_name = self.relative_path+"modelLR/etiquetas.pkl"  

        print("Saving classes labels") 
        with open(model_name, 'wb') as file:  
            pickle.dump(self.image_labels, file)
            
        model_name = self.relative_path+"modelSVM/etiquetas.pkl"  

        with open(model_name, 'wb') as file:  
            pickle.dump(self.image_labels, file)
        
        print("Training finished")
        
        
    def PredictFace(self, img, modelo_name):
        #API Endpoint call this method and send the image was uploaded
        #The same logic is applied. The face is detected, keypoints are extracted and sent to trained ML model to predict.
        #After it the a box is drawn onto the image along class label and % probability of belongs to any class.
        img, detections = self.FindFaceinImage(img=img)
        if detections is not None:
            boxes = self.ExtractFaceRectangle(img,detections,0.4)
            for b in boxes:
                (startX, startY, endX, endY) = b[0]
                points = self.ExtractFacePoints(img[startY:endY,startX:endX])
                if points is not None:
                    result=None
                    etiqueta = None
                    if modelo_name == "SVM":
                        if not self.svc.probability:
                            id_ = self.svc.predict(points)
                            df = self.svc.decision_function(points)
                            result = np.max(softmax(df))
                            etiqueta = self.image_labels_svm[id_[0]]

                        else:
                            id_ = self.svc.classes_[np.argmax(self.svc.predict_proba(points))]
                            result = np.max(self.svc.predict_proba(points))
                            etiqueta = self.image_labels[id_]
                    elif modelo_name == "LR":
                        id_ = self.logistic_r.classes_[np.argmax(self.logistic_r.predict_proba(points))]
                        result = np.max(self.logistic_r.predict_proba(points))
                        etiqueta = self.image_labels_lr[id_]
                            
                    img = self.DrawImageFaceRectangle(img,b[0],result,etiqueta)
            return img


