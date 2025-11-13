#%%
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
Data=pd.read_csv('data.csv')
Data
#%%
Data.isnull().sum()
Data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
#%%
Data.columns

#%%

#%% md
# Train,Test Split
#%%
#Data.iloc[:,1:]get splits into X_Train,X_Test,Abd Data.iloc[:,0] get splits into the y_Train,y_Test
# The train_test_split() function from scikit-learn will automatically split your given data into:
# X_train → training data (features)
# X_test → testing data (features)
# y_train → training labels (target values)
# y_test → testing labels (target values)
X_Train,X_Test,y_Train,y_Test=train_test_split(Data.iloc[:,1:],Data.iloc[:,0],test_size=0.2,random_state=42)
X_Train
#%%

# StandardScaler to make sure all features have equal importance, improve model performance, and ensure faster, stable learning.
scalar=StandardScaler()
X_Train=scalar.fit_transform(X_Train)
X_Test=scalar.transform(X_Test)

#%%
X_Train
X_Test
#%%
y_Train
#%%
encoder=LabelEncoder()
y_Train=encoder.fit_transform(y_Train)
y_Test=encoder.transform(y_Test)

#%%
y_Train
#%% md
# Numpy Array to Pytorch Tensor
# 
#%%
X_Train_Tensor=torch.from_numpy(X_Train)
X_Test_Tensor=torch.from_numpy(X_Test)
y_Train_Tensor=torch.from_numpy(y_Train)
y_Test_Tensor=torch.from_numpy(y_Test)
#%%
X_Train_Tensor.shape
#%%
y_Train_Tensor.shape
#%% md
# 
#%%
print(y_Train_Tensor)
#%%
class MySimpleNN:
    def __init__(self,x):
        self.weights=torch.rand(x.shape[1],1,requires_grad=True,dtype=torch.float64)
        self.bias=torch.zeros(1,dtype=torch.float64,requires_grad=True)

    def forwardPass(self,X):
        z=torch.matmul(X,self.weights)+self.bias
        y_preds=torch.sigmoid(z)
        return y_preds
    def lossFuction(self,y_preds,y):
        epsilon=1e-7
        y_preds=torch.clamp(y_preds,epsilon,1-epsilon)
        loss=-(y_Train_Tensor*torch.log(y_preds)+(1-y_Train_Tensor)*torch.log(1-y_preds)).mean()
        return loss
#%%

#%%
Learning_Rate=0.1
epochs=30
#%%
#Creating the Instance of the ModelClass
Model=MySimpleNN(X_Train_Tensor)
#Define Loop
for epoch in range(epochs):
    y_preds=Model.forwardPass(X_Train_Tensor)
    loss=Model.lossFuction(y_preds,y_Train_Tensor)
    loss.backward()
    with torch.no_grad():
        Model.weights-=Learning_Rate * Model.weights.grad
        Model.bias-=Learning_Rate * Model.bias.grad
    Model.weights.grad.zero_()
    Model.bias.grad.zero_()
    print(f"Epoch :{epoch+1}, and Loss:{loss.item()}")




#%%
Model.bias
#%%
#Model evaluation
with torch.no_grad():
    y_preds=Model.forwardPass(X_Test_Tensor)
    y_preds=(y_preds>0.6).float()
    Accuracy=(y_preds==y_Test_Tensor).float().mean()
print("The Accuracy is ",Accuracy)
