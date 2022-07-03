import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
#  importing the required libraries 
lables = pd.read_excel('C:/mydataset/normal walk/labels.xlsx')
py_list = lables['normal_walk'].tolist()# read the labels file and save it as array
data_array=pd.read_excel('C:/mydataset/normal walk/thenew method .xlsx') #read the simplified dataset 
X_train, X_test, y_train, y_test = train_test_split(data_array, py_list, test_size=0.1, random_state=42)
norm=Normalizer()
scaler = StandardScaler()
clf= KNeighborsClassifier(n_neighbors=5)
pipeline = make_pipeline( norm,scaler,clf)
pipeline.fit(X_train,y_train)
predict = pipeline.predict(X_test)
print(predict)
accuracy=pipeline.score(X_test,y_test)
print(accuracy)
conf=confusion_matrix(y_test, predict,labels=["normal_walk", "ramp_ascent", "stair_ascent","ramp_descnt","stair_descnt"])
print(conf)
initial_type = [('float_input', FloatTensorType([1, 12]))]
onx = convert_sklearn(clf, initial_types=initial_type)
with open("NewMethodeClassfire.onnx", "wb") as f:
    f.write(onx.SerializeToString())  # save the classifier as ONNX format file 
