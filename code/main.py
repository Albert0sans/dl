from datetime import datetime

from utils.window_generator import WindowGenerator
from utils.preproces import trainTestSplit, preprocesDf,computeFeatures
import utils.layers.dl_training as DL
import utils.multi_class_training as mc
import utils.dl_layers as CustomLayers
from  utils.dl_layers import AutoregressiveWrapperLSTM,GenerativeAdversialEncoderWrapper
from utils.plotting import multiModelComparison
import matplotlib.pyplot as plt
from utils.algos import simple_backtest
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('float16')
mixed_precision.set_global_policy(policy)
import seaborn as sns
pd.set_option('display.max_columns', None)
physical_devices = tf.config.list_physical_devices()
print(f"Physical devices: {physical_devices}")

import vectorbt as vbt





def simplebacktest(y_true,y_pred)-> tuple[float, float, float, float, float]:
    print(np.shape(y_pred))
    print(np.shape(y_true))
    entries=y_pred > 0.0
    exits=y_pred < -0.0
    y_true = np.exp(y_true.cumsum()) 

    test=(y_true*entries).cumsum()

    pf = vbt.Portfolio.from_signals(y_true, entries, exits, init_cash=100,fees=0.001,sl_stop=0.05,tp_stop=0.1)
    stats = pf.stats(silence_warnings=True) # Add silence_warnings=True here


    win_rate=stats["Win Rate [%]"]
    avg_losing=stats["Avg Losing Trade [%]"]
    avg_winning=stats["Avg Winning Trade [%]"]
    benchmark_return=stats["Win Rate [%]"]
    benchmark_return=stats["Benchmark Return [%]"]
    total_return=stats["Total Return [%]"]
    
    return benchmark_return,total_return,avg_losing,avg_winning,win_rate

price_target="close IVV"


OUT_STEPS = 1
INPUT_WIDTH=10000
MAX_EPOCHS = 200
BATCH_SIZE=32
output={}
df = pd.read_csv("financial_data2.csv",header=[0,1],index_col=0 )
df=df.drop('VIXM', axis=1, level=1)
print(df.isnull().sum())
df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

df.columns = [' '.join(col).strip() for col in df.columns.values]
df=df.dropna()

df=df.sample(frac=0.2).sort_index()
print(len(df))
# Shuffle the DataFrame

# Split into 80/20
split_idx = int(len(df) * 1)
validation_df = df[split_idx:]

df = df[:split_idx]

df = preprocesDf(df,price_target)
print(np.shape(df))
n=len(df)

source_cols=df.columns
label_columns=[col for col in df.columns if col.startswith('target')]
source_cols_notarget=[col for col in df.columns if not col.startswith('target')]
in_features=len(source_cols_notarget)

df=df.dropna()
num_features = len(label_columns)

source_cols=df.columns




train_df, test_df, val_df,mean,std = trainTestSplit(df,label_columns)



mean=mean.values.reshape(1,OUT_STEPS,num_features)
std=std.values.reshape(1,OUT_STEPS,num_features)


multi_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=OUT_STEPS,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    shift=1,
    label_columns=label_columns,
)



del df,train_df,val_df,test_df


train_generator=multi_window.train



test_generator=multi_window.test
val_generator=multi_window.val





discriminator=CustomLayers.discriminator(OUT_STEPS, num_features)
generator=CustomLayers.generator(32,OUT_STEPS, num_features)


models = {
  #  "test":CustomLayers.ZeroBaseline(OUT_STEPS,num_features),
   # "informer":informer,
 #"transformer":CustomLayers.transformer(INPUT_WIDTH,in_features,OUT_STEPS,num_features),
  "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
  # "gan":GenerativeAdversialEncoderWrapper(OUT_STEPS=OUT_STEPS,generator=generator,discriminator=discriminator, num_features=in_features,),
  # "ar_lstmstatefull_model":AutoregressiveWrapperLSTM(OUT_STEPS= OUT_STEPS,num_features=in_features),
  #  "auto_encoder":CustomLayers.auto_encoder(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   
  #  "cnn":CustomLayers.cnn_layer(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   

 # "rnn_model": CustomLayers.rnn_model(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #  "rnn_model_gru": CustomLayers.rnn_model_gru(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #"extrarfsklearn":CustomLayers.extrarf(OUT_STEPS,num_features),
 #"cnnlstm1":CustomLayers.cnnlstm1(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   
 # "cnnlstm2":CustomLayers.cnnlstm2(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   
 #"cnnlstm3":CustomLayers.cnnlstm3(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   

#"ydf":CustomLayers.gbt(OUT_STEPS,num_features),
#"hgb":CustomLayers.hgb(OUT_STEPS,num_features),
#  "rfsklearn":CustomLayers.rf(OUT_STEPS,num_features),
}

# validate model adjusts to shape
for name,model in models.items():
    if hasattr(model, "build") and callable(getattr(model, "build")):
        inputshape=(None,INPUT_WIDTH,num_features)
       # model.build(input_shape=inputshape)  # batch size is None (flexible)


model_metrics = {}


for name, model in models.items():
    print(name)
    models[name]=mc.MultiClassModel(
                        model=model,
                        retrain=True,
                         model_name=name,
                         epochs=MAX_EPOCHS,
                         target_indices=multi_window.target_indices,
                      
                             )
   

    
    history=models[name].fit(
          train_generator_fn=lambda:multi_window.train,
          val_generator_fn=lambda:multi_window.val
    ) 
    
    history=history.history
    
    
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
   # plt.show()
    metrics = models[name].evaluate(lambda:multi_window.test)
    model_metrics[name]=metrics

# Plot all fitted models

#multiModelComparison(list(model_metrics.values()), list(models.keys()))


#multi_window.plot(list(models.values()), plot_col=target_cols[0])

# First, get all true y values from the generator
   

# The rest remains the same
for name, model in models.items():
    test_generator=multi_window.test
    y_test_batches = []
    for x,y in test_generator:  # assuming test_generator has __len__ defined
        
        y_test_batches.append(y)
    
    y_test = np.concatenate(y_test_batches)
    # Predict using the generator without loading all X_test at once
    forecast = models[name].predict(lambda:multi_window.test)
    # Now apply inverse scaling
    true = (y_test * std + mean)
    forecast = (forecast * std + mean)

    plt.plot(true.flatten())
    plt.plot(forecast.flatten())


    df = pd.DataFrame({"forecast": forecast.flatten(), "real": true.flatten()})

    benchmark_return, total_return, avg_losing, avg_winning, win_rate = simplebacktest(true, forecast[:, :, 0])

    model_metrics[name]["total_return"]= total_return

    

    print(f"""
    Strategy Evaluation: {name}
    ---------------------------------------
    Benchmark Return : {benchmark_return}
    Total Return     : {total_return}
    Win Rate         : {win_rate}
    Avg Winning Trade: {avg_winning}
    Avg Losing Trade : {avg_losing}
    ---------------------------------------
    """)

fig, axes = plt.subplots(2, 1, ) # Increased figure width for better readability
print(model_metrics)


model_names=list(models.keys())
total_return_values = [model_metrics[model]['total_return'] for model in model_names]

# Create a DataFrame for the remaining metrics
remaining_metrics_data = []
for model_name, metrics in model_metrics.items():
    for metric_name, value in metrics.items():
        if metric_name != 'total_return':
            remaining_metrics_data.append({'Model': model_name, 'Metric': metric_name, 'Value': value})

df_remaining_metrics = pd.DataFrame(remaining_metrics_data)
# Bar plot for Total Return
sns.barplot(x=model_names, y=total_return_values, ax=axes[1])
axes[1].axhline(y=benchmark_return, color='red', linestyle='--', linewidth=1.5, label=f'Benchmark ({benchmark_return:.2f})')
axes[1].set_title('Total Return for Each Model (Backtest)')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Total Return')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].legend()

# Bar plot for remaining metrics
# Using seaborn.catplot for easier facet plotting of multiple metrics on one graph
sns.barplot(data=df_remaining_metrics, x='Model', y='Value', hue='Metric', ax=axes[0], palette='viridis')
axes[0].set_title('Other Model Metrics')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Value')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].legend(title='Metric')

plt.tight_layout()
plt.show()




