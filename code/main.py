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
import seaborn as sns
pd.set_option('display.max_columns', None)

import vectorbt as vbt

def simplebacktest(real:pd.DataFrame,entries:pd.DataFrame,exits:pd.DataFrame)-> tuple[float, float, float, float, float]:
    
    real = np.exp(real.cumsum()) 
    
    pf = vbt.Portfolio.from_signals(real, entries, exits, init_cash=100,fees=0.001,sl_stop=0.05,tp_stop=0.1)
    stats = pf.stats(silence_warnings=True) # Add silence_warnings=True here
    win_rate=stats["Win Rate [%]"]
    avg_losing=stats["Avg Losing Trade [%]"]
    avg_winning=stats["Avg Winning Trade [%]"]
    benchmark_return=stats["Win Rate [%]"]
    benchmark_return=stats["Benchmark Return [%]"]
    total_return=stats["Total Return [%]"]
    return benchmark_return,total_return,avg_losing,avg_winning,win_rate

OUT_STEPS = 1
INPUT_WIDTH=180
MAX_EPOCHS = 100
BATCH_SIZE=32
output={}

df = pd.read_csv("financial_data2.csv",header=[0,1],index_col=0 )
df=df.drop('VIXM', axis=1, level=1)
print(df.isnull().sum())
df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

df.columns = [' '.join(col).strip() for col in df.columns.values]
df=df.dropna()

df=df.sample(frac=1).sort_index()
print(len(df))
# Shuffle the DataFrame

# Split into 80/20
split_idx = int(len(df) * 1)
validation_df = df[split_idx:]

df = df[:split_idx]

df = preprocesDf(df)

n=len(df)

source_cols=df.columns
target_cols=["close IVV"]
df = df.rename(columns={target_cols[0]: 'targets',})

df=df.dropna()
num_features = len(target_cols)
in_features=len(df.columns)
source_cols=df.columns


print(df.head(4))


plt.show()


train_df, test_df, val_df,mean,std = trainTestSplit(df)

multi_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=OUT_STEPS,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    shift=2,
    label_columns=["targets"],
)


X_train,y_train=multi_window.train
real = np.exp(y_train.flatten().cumsum()) 


X_test,y_test=multi_window.test
X_val,y_val=multi_window.val




discriminator=CustomLayers.discriminator(OUT_STEPS, num_features)
generator=CustomLayers.generator(32,OUT_STEPS, num_features)


models = {
  #  "test":CustomLayers.ZeroBaseline(OUT_STEPS,num_features),
  #  "informer":informer,
  #"transformer":CustomLayers.transformer(INPUT_WIDTH,in_features,OUT_STEPS,num_features),
  #"multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
  # "gan":GenerativeAdversialEncoderWrapper(OUT_STEPS=OUT_STEPS,generator=generator,discriminator=discriminator, num_features=in_features,),
  # "ar_lstmstatefull_model":AutoregressiveWrapperLSTM(OUT_STEPS= OUT_STEPS,num_features=in_features),
  #  "auto_encoder":CustomLayers.auto_encoder(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   
  #  "cnn":CustomLayers.cnn_layer(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   

 # "rnn_model": CustomLayers.rnn_model(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #  "rnn_model_gru": CustomLayers.rnn_model_gru(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #"extrarfsklearn":CustomLayers.extrarf(OUT_STEPS,num_features),
 "ydf":CustomLayers.gbt(OUT_STEPS,num_features),
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
                         X_train=X_train,y_train=y_train,
                         X_test=X_test,y_test=y_test,
                         X_val=X_val,y_val=y_val,
                
                        
                             )
   


    models[name].fit() 
    metrics = models[name].evaluate()
    model_metrics[name]=metrics

# Plot all fitted models

#multiModelComparison(list(model_metrics.values()), list(models.keys()))


#multi_window.plot(list(models.values()), plot_col=target_cols[0])




fig, axes = plt.subplots(2, len(models),  )



for name, model in models.items():

    forecast = model.predict(X_test).flatten()
    true = y_test*std+mean
    forecast=forecast*std+mean
    if False:
        ax_forecast = axes[0, idx]
        ax_kde = axes[1, idx]
        ax_forecast.plot(forecast.flatten(), label="Forecast")
        ax_forecast.plot(true.flatten(), label="Real",alpha=0.3)
        ax_forecast.set_title(key)
        ax_forecast.legend()
        ax_forecast.grid(True)

        sns.kdeplot(forecast.flatten(), bw_adjust=0.5, ax=ax_kde,label="Forecast")
        sns.kdeplot(true.flatten(), bw_adjust=0.5, ax=ax_kde,label="Real")

        ax_kde.set_title(f"KDE - {key}")
        ax_kde.set_xlabel("Value")
        ax_kde.set_ylabel("Density")
        ax_kde.grid(True)
        print(f"{key} - mse: {model_metrics[key]['mse']:.6f}, r2: {model_metrics[key]['r2']:.6f}")
        # Optional backtest dataframe
    df = pd.DataFrame({"forecast": forecast.flatten(), "real": true.flatten()})

    

    entries=forecast > 0.0
    exits=forecast < -0.0

    benchmark_return,total_return,avg_losing,avg_winning,win_rate=simplebacktest(real=true.flatten(),entries=entries,exits=exits,)
    if name not in output:
        output[name] = {"r2": [], "total_return": []} # Initialize with empty lists if new

        # Append the current R2 and total_return to their respective lists
    output[name]["r2"].append(model_metrics[name]['r2'])
    output[name]["total_return"].append(total_return)
    if False:
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

  # Extract data for plotting
model_names = []
r2_values = []
total_return_values = []

for model_name, metrics in output.items():
    model_names.append(model_name)
    # Extract the single value from the list
    r2_values.append(metrics['r2'][0])
    total_return_values.append(metrics['total_return'][0]) 

fig, axes = plt.subplots(2, 1, ) # Increased figure width for better readability

# Bar plot for R2 scores
sns.barplot(x=model_names, y=r2_values, ax=axes[0], palette='viridis')
axes[0].set_title('R2 Score for Each Model')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('R2 Score')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Bar plot for Total Return
sns.barplot(x=model_names, y=total_return_values, ax=axes[1], palette='plasma')
axes[1].axhline(y=benchmark_return, color='red', linestyle='--', linewidth=1.5, label=f'Benchmark ({benchmark_return})')
axes[1].set_title('Total Return for Each Model (Backtest)')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Total Return')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout() # Adjust layout to prevent overlapping elements
