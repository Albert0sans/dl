from utils.window_generator import WindowGenerator
from utils.preproces import trainTestSplit, preprocesDf
#import utils.dl_training as DL
from utils.layers.transformer import transformer_timeseries
import utils.multi_class_training as mc
import utils.dl_layers as CustomLayers
from  utils.dl_layers import AutoregressiveWrapperLSTM,GenerativeAdversialEncoderWrapper
from utils.plotting import multiModelComparison
from utils.layers.informer.informer import Informer
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf



OUT_STEPS = 31
INPUT_WIDTH=360
MAX_EPOCHS = 500
BATCH_SIZE=32

df = pd.read_csv("dataset.csv")
df=df.sample(frac=1).sort_index()
df = preprocesDf(df)





target_cols = ["TARGET_close_AAPL"]
num_features = len(target_cols)
in_features=len(df.columns)
source_cols=df.columns

train_df, test_df, val_df = trainTestSplit(df)
multi_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=OUT_STEPS,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    shift=OUT_STEPS,
    label_columns=target_cols,
)




X_train,y_train=multi_window.train
X_test,y_test=multi_window.test
X_val,y_val=multi_window.val


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)




discriminator=CustomLayers.discriminator(OUT_STEPS, num_features)
generator=CustomLayers.generator(32,OUT_STEPS, num_features)

transformer=transformer_timeseries(
 
    input_shape= (INPUT_WIDTH,in_features),
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,

    output_size=[OUT_STEPS,num_features]
)



informer = Informer(
            enc_in= 7,
    dec_in= 7,
    c_out= 7,
    seq_len= INPUT_WIDTH,
    label_len= INPUT_WIDTH,
    out_len= OUT_STEPS,
    batch_size=BATCH_SIZE,
    factor= 5,
    d_model= 512,
    n_heads= 8,
    e_layers= 3,
    d_layers= 2,
    d_ff= 512,
dropout= 0,
    attn= 'prob',
    embed= 'fixed',
    data= 'ETTh',
    activation= 'gelu'

        )
if True:
    informer.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mae, metrics=['mse', 'mae'],
                            run_eagerly=True)
    informer.fit(train_ds,
                        steps_per_epoch=20,
                        
                        validation_steps=20,
                        
                        epochs=10)

exit(0)
models = {
  #  "informer":informer,
   "transformer":transformer,
  "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
     "gan":GenerativeAdversialEncoderWrapper(OUT_STEPS=OUT_STEPS,generator=generator,discriminator=discriminator, num_features=in_features,),
    "ar_lstmstatefull_model":AutoregressiveWrapperLSTM(OUT_STEPS= OUT_STEPS,num_features=in_features),
    "auto_encoder":CustomLayers.auto_encoder(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
    #"transformer_model":CustomLayers.random_forest(OUT_STEPS=OUT_STEPS, num_features=num_features),
   
    "rnn_model": CustomLayers.rnn_model(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
    "rnn_model_gru": CustomLayers.rnn_model_gru(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
    "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
   # "extrarfsklearn":CustomLayers.extrarf(OUT_STEPS,num_features),

    "rfsklearn":CustomLayers.rf(OUT_STEPS,num_features),
}
{

}

# validate model adjusts to shape
for name,model in models.items():
    if hasattr(model, "build") and callable(getattr(model, "build")):
        inputshape=(None,INPUT_WIDTH,num_features)
       # model.build(input_shape=inputshape)  # batch size is None (flexible)







model_metrics = []



for name, model in models.items():
    model=mc.MultiClassModel(
                        model=model,
                        retrain=True,
                         model_name=name,
                         epochs=MAX_EPOCHS,
                         target_indices=multi_window.target_indices,
                         X_train=X_train,y_train=y_train,
                         X_test=X_test,y_test=y_test,
                         X_val=X_val,y_val=y_val,
                         train_ds=train_ds,
                         test_ds=test_ds,  
                         val_ds=val_ds,
                       
                      

                             )
   
    models[name] = model.fit() 
    metrics = model.evaluate()
    model_metrics.append(metrics)

# Plot all fitted models

multiModelComparison(model_metrics, list(models.keys()))


multi_window.plot(list(models.values()), plot_col=target_cols[0])


