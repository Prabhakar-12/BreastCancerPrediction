#%%
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
#%%
Data=pd.read_csv('data.csv')
Data.isnull().sum()
#%%
Data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
#%%
print(Data.columns)
display(Data)
#%%
#Spliting the Dataset To Train and Test
X_Train,X_Test,y_Train,y_Test=train_test_split(Data.iloc[:,1:],Data.iloc[:,0],test_size=0.2,random_state=42)
display(len(X_Train))
display(len(y_Train))
X_Train
#%%
#StandardScalar:For arrangeing all neumerical Values in the same Scale for informign to the model all features should be equal Priority
scalar=StandardScaler()
X_Train=scalar.fit_transform(X_Train)
X_Test=scalar.transform(X_Test)
X_Train
#%%
Encoder=LabelEncoder()
y_Train=Encoder.fit_transform(y_Train)
y_Test=Encoder.transform(y_Test)
y_Train
#%% md
# 
#%%

#%% md
# 
#%% md
# 
#%% md
# 
#%%
#Converting the Numpy Array to Python Tensors
X_Train_Tensor=torch.from_numpy(X_Train)
X_Test_Tensor=torch.from_numpy(X_Test)
y_Train_Tensor=torch.from_numpy(y_Train)
y_Train_Tensor=torch.from_numpy(y_Train)
#%%
class MySimpleNN:
    def __init__(self,x):
        self.weights=torch.rand(x.shape[1],1,require_grad=True,dtype=torch.float64)
        self.bias=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    def forward(self,X):
        z=torch.matmul(X,self.weights)+self.bias
        y_preds=torch.sigmoid(z)
    def LossFunction(self,y_preds,y):
        epsilon=1e-7
        y_preds=torch.clamp(y_preds,epsilon,1-epsilon)
        loss=-(y_Train_Tensor+torch.log(y_Train_Tensor)+(1-y_Train_Tensor)+(torch.log(1-y_Train_Tensor))).mean()
        return loss

#%%
Learning_Rate=0.1
epochs=30

#%%
#Creating the Instance of the ModelClass
Model=MySimpleNN(X_Train_Tensor)
#Define Loop
for epoch in range(epochs):
    y_preds=Model.forward(X_Train_Tensor)
    loss=Model.LossFunction(y_preds,y_Train_Tensor)
    loss.backward()
    with torch.no_grad():
        Model.weights-=Learning_Rate*Model.weights.grad
        Model.bias-=Learning_Rate*Model.bias.grad
    Model.weights.grad.zero_()
    Model.bias.grad.zero_()
    print(f"Epoch :{epoch+1},and Loss :{loss.item()}")
#%%
