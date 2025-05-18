import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from pickle import dump,load
import os

def fit_keras(model,train_dataset,validation_dataset,model_path,epochs):
    model.compile(
                loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()],
            )
            
    early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                mode="min",
                restore_best_weights=True
            )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,  # or use 'best_model.keras' for the new format
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,  # Set to True if you only want weights
                mode="min",
                verbose=1
            )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor  =0.1,
                                                    patience=3,
                                                    min_lr=0.00001)

    model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[early_stopping,checkpoint,reduce_lr],
            )
    return model
def fit_sklearn(model,X_train,y_train):
    return model.fit(X_train,y_train)
def fit_ydf(model,train_dict):
    return model.train(train_dict)

def get_data_for_keras( X_train,y_train,X_test,y_test,X_val,y_val):

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    return train_ds,test_ds,val_ds
def get_data_for_ydf(X_train,y_train,source_cols,target_cols):
    X_train_2d = X_train.reshape(X_train.shape[0], -1)  # (n_samples, timesteps × features)
    y_train_2d = y_train.reshape(y_train.shape[0], -1)  # (n_samples, timesteps × features)
    train_ds_dicts = {
        **dict(zip(source_cols, X_train_2d.T)),
        **dict(zip(target_cols, y_train_2d.T if y_train_2d.ndim > 1 else [y_train_2d]))
    }
    return train_ds_dicts
    
    
tf.keras.config.enable_unsafe_deserialization()

def load_keras_model(model_path):
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
       
        new_model = tf.keras.models.load_model(model_path)
       
        return new_model
    return False

def save_keras_model(model,model_path):
    
def save_pickle_model(model,model_path):
    with open(model_path,"wb") as f:
        dump(model,model_path,protocol=5)

def load_pickle_model(name):
    model_path = f"./models/{name}.pkl"
    if os.path.exists(model_path):
        with open(model_path,"rf") as f:
                return load(f)

def eval(model, test_dataset):
    # Evaluate the model on the test dataset
    
    names=[]
    if hasattr(model, "evaluate") and callable(getattr(model, "evaluate")):
    
    
        result = model.evaluate(test_dataset, verbose=0)  # Suppress verbose output
        names=model.metrics_names
    else:
        predictions=model.predict(test_dataset(0))
        mse = mean_squared_error(test_dataset(1), predictions)
        print(f'Mean Squared Error: {mse}')

        r2 = r2_score(test_dataset(1), predictions)
        print(f'R-squared: {r2}')
        result=[mse,r2]
        names=["mse","r2"]
    if not isinstance(result, list):
        result = [result]
    # Map metric names to their corresponding values
    metrics_dict = dict(zip(names, result))

    return metrics_dict

    
        

   

