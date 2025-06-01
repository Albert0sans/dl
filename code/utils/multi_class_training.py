import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from pickle import dump,load
import os
from scipy.stats import ttest_ind
from ydf import GenericLearner
import ydf
from tensorflow.keras import Model as KerasModel
from sklearn.base import BaseEstimator
tf.keras.config.enable_unsafe_deserialization()
import numpy as np

class MultiClassModel:
    def __init__(
        self,
        model,
        model_name,
        target_indices,
        retrain=False,
        epochs=10,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,

        
        
    ):
        self.retrain=retrain
        self.model=model
        self.model_name=model_name
        self.target_indices=target_indices
        self.epochs=epochs
        assert isinstance(self.target_indices, list) and len(self.target_indices) < X_train.shape[-1]

        self.mode_type= self.getType()
        assert self.mode_type != False 
        loss_magnitude = np.abs(np.minimum(y_train, 0))  # Only penalize negative returns
        sample_weights = 1.0 + (loss_magnitude / loss_magnitude.max()) * 4.0  # Scale up to 5x
        self.sample_weights=sample_weights.flatten()
        assert (X_train is not None) & (y_train is not None) & (X_test is not None) & (y_test is not None)  & (X_val is not None) & (y_val is not None) 
        match self.mode_type:
            case "keras":
                self.model_path = f"./models/{model_name}.keras"
                        # Calculate sample weights only for training set
                


                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, self.sample_weights)).batch(32)
                val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
                self.train_ds=train_ds
                self.test_ds=test_ds
                self.val_ds=val_ds
            case "sklearn":
               
                self.model_path = f"./models/{model_name}.pkl"

            case "ydf":
               
                self.model_path = f"./models/{model_name}.pkl"

                train_dict=make_ds_dict(X=X_train,y=y_train,)
                test_dict=make_ds_dict(X=X_train,y=y_train)
                val_dict=make_ds_dict(X=X_train,y=y_train)
                self.train_dict=train_dict
                self.test_dict=test_dict
                self.val_dict=val_dict
                 


        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.X_val=X_val
        self.Y_val=y_val


        


        
        


        
    def runforeachclass(
        self,
        keras_code,
        sklearn_code,
        ydf_code
    ):
         match self.mode_type:
            
            case "keras":
               return  keras_code()
            case "sklearn":
               return sklearn_code()
            case "ydf":
               return ydf_code()
    def fit(self,):
        sample_weight=self.sample_weights
        self.runforeachclass(
           keras_code=lambda: self.fit_keras(
                               
                              ),
           sklearn_code=lambda: self.fit_sklearn(self.model,self.X_train,self.y_train,sample_weight=sample_weight),
           ydf_code=lambda: self.fit_ydf(self.model,self.train_dict,sample_weight=sample_weight)
       )
        return self.model
     
    def predict(self,data):
        
        
        pred = self.runforeachclass(
           keras_code=lambda: self.predict_keras(data
                               
                              ),
           sklearn_code=lambda: self.predict_sklearn(data),
           ydf_code=lambda: self.predict_ydf(data)
       )
        return pred
        
     
    def evaluate(self,test_dataset=None):
        # Evaluate the model on the test dataset
        
        predictions=None
        true_vals=None
        match self.mode_type:
            case "keras":
                test_dataset=self.test_ds
                predictions=self.predict(test_dataset)
                true_vals=self.y_test
                
            case "sklearn":
                test_dataset=(self.X_test,self.y_test)
                predictions=self.predict(test_dataset[0])
                true_vals=test_dataset[1]
                
            case "ydf":
                test_dataset=(self.X_test,self.y_test)
                predictions=self.predict(test_dataset[0])
                true_vals=test_dataset[1]
 

        mse = np.mean((true_vals- predictions) ** 2)
      
        try:
            r2 = r2_score(true_vals.flatten(), predictions.flatten())
        except:
            r2=0
       
 
        p2=ttest_ind(true_vals.flatten(), predictions.flatten()).pvalue
        

        result=[mse,r2,p2]
        names=["mse","r2","p2"]
        if not isinstance(result, list):
            result = [result]
        # Map metric names to their corresponding values
        metrics_dict = dict(zip(names, result))

        return metrics_dict
        
    def fit_sklearn(self,model,X_train,y_train, **kwargs  ):
        if(self.retrain is False):
            model=load_pickle_model(self.model_path)
            if(model is not False):
                self.model=model
                return self.model
        

        y_train=y_train.ravel()
        self.model.fit(X_train,y_train, **kwargs  )
        save_pickle_model(self.model,self.model_path)
           
        
        return self.model
    def fit_ydf(self,model,train_dict, **kwargs ):
        if(self.retrain is False):
            if(os.path.exists(self.model_path)):
                model=ydf.load_model(self.model_path)
                if(model is not False):
                    self.model=model
                    return self.model

        self.model= model.train(train_dict)
        self.model.save(self.model_path)
    def getType(self,):
        originalmodelinstance=self.model
        if(hasattr(self.model,"model")):
            originalmodelinstance=self.model.model
        if isinstance(originalmodelinstance, KerasModel):
            return "keras"
        elif isinstance(originalmodelinstance, BaseEstimator):
            return "sklearn"
        elif isinstance(originalmodelinstance, GenericLearner):
            return "ydf"
        else:
            return False
        
    def fit_keras(self,**kwargs):
        model=False
        if(self.retrain is False):
            model= load_keras_model(model_path=self.model_path)
        if(model is not False):
            self.model=model
            return
        self.model.compile(
                    loss=tf.losses.Huber(),
                    optimizer=tf.optimizers.Adam(learning_rate=1e-4,clipnorm=1.0),
                    metrics=[tf.metrics.MeanAbsoluteError()],
                )
                
        early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=6,
                  
                    restore_best_weights=True
                )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.model_path,  # or use 'best_model.keras' for the new format
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,  # Set to True if you only want weights
                 
                    verbose=0
                )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor  =0.1,
                                                        patience=3,
                                                        min_lr=0.00001)

        self.model.fit(
                    self.train_ds,
                    epochs=self.epochs,
                    validation_data=self.val_ds,
                    callbacks=[early_stopping,checkpoint,reduce_lr],
                    verbose=1,
                    steps_per_epoch=10,
                     **kwargs  
                    
                )
        return self.model
        



    def predict_ydf(self,data):
        
        if (not isinstance(data,dict)):
            data=make_ds_dict(X=data)
            print(data)
            pred=self.model.predict(data)
        else:
            pred=self.model.predict(data)
        return pred
    def predict_keras(self,data):
        pred=self.model.predict(data,verbose=0)
        if(pred.shape[-1]>len(self.target_indices)):
            try:
                return pred[:,:,self.target_indices]
            except:
                pass
        return pred
    def predict_sklearn(self,data):
        pred=self.model.predict(data)
        if(pred.shape[-1]>len(self.target_indices)):
            try:
                return pred[:,:,self.target_indices]
            except:
                pass
        return pred


def get_data_for_keras( X_train,y_train,X_test,y_test,X_val,y_val):

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    return train_ds,test_ds,val_ds
def make_ds_dict(X, y=None):
        print(np.shape(X))
        data_dict = {
       
        f"{i}": X[:, :, i] for i in range(X.shape[2])
        }
        if(y is not None):
            data_dict["targets"] = y.flatten()

        return data_dict
def get_data_for_ydf(X_train, y_train, X_test, y_test, X_val, y_val, source_cols, target_cols):
    
    train_ds_dicts,test_ds_dicts,val_ds_dicts=None
    if(X_train and y_train):
        train_ds_dicts = make_ds_dict(X_train, y_train)
    if(X_test and y_test):
        test_ds_dicts = make_ds_dict(X_test, y_test)
    if(X_val and y_test):
        val_ds_dicts = make_ds_dict(X_val, y_val)

    return train_ds_dicts, test_ds_dicts, val_ds_dicts

    
    
tf.keras.config.enable_unsafe_deserialization()

def load_keras_model(model_path):
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
       
        new_model = tf.keras.models.load_model(model_path)
       
        return new_model
    return False

    
    
def save_pickle_model(model,model_path):
    with open(model_path,"wb+") as f:
        dump(model,f,protocol=5)

def load_pickle_model(model_path):
    
    if os.path.exists(model_path):
        with open(model_path,"rb") as f:
                return load(f)
    return False
