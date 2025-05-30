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



OUT_STEPS = 1
INPUT_WIDTH=180
MAX_EPOCHS = 10
BATCH_SIZE=32

df = pd.read_csv("download.csv",index_col=0,header=[0, 1] )
df.columns = [' '.join(col).strip() for col in df.columns.values]
df=df.dropna()
print(df.head())
df=df.sample(frac=1).sort_index()
# Shuffle the DataFrame

# Split into 80/20
split_idx = int(len(df) * 1)
validation_df = df[split_idx:]

df = df[:split_idx]

df = preprocesDf(df)
#df=computeFeatures(df)
n=len(df)



source_cols=df.columns
target_cols=["Close SPY"]
df = df.rename(columns={target_cols[0]: 'targets',})

df=df.dropna()
num_features = len(target_cols)
in_features=len(df.columns)
source_cols=df.columns


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

print(np.isnan(df).any(), np.isinf(df).any())

print(np.isnan(X_train).any(), np.isinf(X_train).any())
print(np.isnan(y_train).any(), np.isinf(y_train).any())
X_test,y_test=multi_window.test
X_val,y_val=multi_window.val



train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)


discriminator=CustomLayers.discriminator(OUT_STEPS, num_features)
generator=CustomLayers.generator(32,OUT_STEPS, num_features)


models = {
  #  "test":CustomLayers.ZeroBaseline(OUT_STEPS,num_features),
  #  "informer":informer,
  "transformer":CustomLayers.transformer(INPUT_WIDTH,in_features,OUT_STEPS,num_features),
  "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
    "gan":GenerativeAdversialEncoderWrapper(OUT_STEPS=OUT_STEPS,generator=generator,discriminator=discriminator, num_features=in_features,),
   "ar_lstmstatefull_model":AutoregressiveWrapperLSTM(OUT_STEPS= OUT_STEPS,num_features=in_features),
    "auto_encoder":CustomLayers.auto_encoder(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   
     "cnn":CustomLayers.cnn_layer(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),   

 # "rnn_model": CustomLayers.rnn_model(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #  "rnn_model_gru": CustomLayers.rnn_model_gru(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
   #"extrarfsklearn":CustomLayers.extrarf(OUT_STEPS,num_features),
 "ydf":CustomLayers.gbt(OUT_STEPS,num_features),
"hgb":CustomLayers.hgb(OUT_STEPS,num_features),
   "rfsklearn":CustomLayers.rf(OUT_STEPS,num_features),
}

# validate model adjusts to shape
for name,model in models.items():
    if hasattr(model, "build") and callable(getattr(model, "build")):
        inputshape=(None,INPUT_WIDTH,num_features)
       # model.build(input_shape=inputshape)  # batch size is None (flexible)


model_metrics = {}



for name, model in models.items():
    models[name]=mc.MultiClassModel(
                        model=model,
                        retrain=False,
                         model_name=name,
                         epochs=MAX_EPOCHS,
                         target_indices=multi_window.target_indices,
                         X_train=X_train,y_train=y_train,
                         X_test=X_test,y_test=y_test,
                         X_val=X_val,y_val=y_val,
                         train_ds=train_ds,
                         test_ds=test_ds,  
                         val_ds=val_ds,
                         train_dict=mc.make_ds_dict(X=X_train,y=y_train,),
                        test_dict=mc.make_ds_dict(X=X_train,y=y_train),
                        val_dict=mc.make_ds_dict(X=X_train,y=y_train),
                             )
   
    models[name].fit() 
    metrics = models[name].evaluate()
    model_metrics[name]=metrics

# Plot all fitted models

#multiModelComparison(list(model_metrics.values()), list(models.keys()))


#multi_window.plot(list(models.values()), plot_col=target_cols[0])




fig, axes = plt.subplots(2, len(models),  )
import vectorbt as vbt

def simplebacktest(real:pd.DataFrame,entries:pd.DataFrame,exits:pd.DataFrame)-> tuple[float, float]:
    
    real = np.exp(real.cumsum()) 
    
    pf = vbt.Portfolio.from_signals(real, entries, exits, init_cash=100,fees=0.001,sl_stop=0.05,tp_stop=0.1)
    stats = pf.stats(silence_warnings=True) # Add silence_warnings=True here
    print(stats)
    benchmark_return=stats["Benchmark Return [%]"]
    total_return=stats["Total Return [%]"]
    return benchmark_return,total_return
    

for idx, key in enumerate(model_metrics):
    model = models[key]
    forecast = model.predict(X_test).flatten()
    true = y_test*std+mean
    forecast=forecast*std+mean
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

    benchmark_return,total_return=simplebacktest(real=true.flatten(),entries=entries,exits=exits,)
    
    print(f"{key} benchmark returns {benchmark_return:0.3f}, total returns {total_return:0.3f}")




   


plt.show()

exit()

def process_row(previous_row,current_row):
     
     return np.log(previous_row /current_row).values

def simulate_trading(model, val_df, input_width, target_col="target", threshold=0.000,mean=None,std=None):
    """
    Simulates a simple trading strategy using model predictions.
    Decision: Buy if predicted return > +threshold, Sell if < -threshold, else Hold.
    
    Returns:
        - DataFrame with datetime, prediction, actual, action, returns, cumulative return.
    """

    current_model_input = []
    records = []

    for i in range(1, len(val_df) - 2):  # -1 to avoid index overflow
        previous_row = val_df.iloc[i - 1]
        current_row = val_df.iloc[i]
        
        # Compute log change
        log_change = process_row(previous_row, current_row)
        if mean is not None and std is not None:
           log_change = (log_change - mean) / std

        current_model_input.append(log_change)

        # Maintain window size
        if len(current_model_input) > input_width:
            current_model_input.pop(0)

        if len(current_model_input) == input_width:
            
            
            
            # Prepare input
            model_input = np.vstack(current_model_input)[np.newaxis, :, :]
            model_input = np.nan_to_num(model_input,0)
            prediction = model.predict(model_input)
            
            prediction=prediction.flatten()  # Assume single float predicted: return or price delta
       
            # Compute actual return (log return)
            actual = np.log(val_df.iloc[i + 2][target_col] / current_row[target_col])
            
            # Simple threshold-based decision
            if prediction > threshold:
                action = "buy"
                realized_return = actual
            else:
                action = "hold"
                realized_return = 0

            records.append({
             #   "datetime": val_df.iloc[i + 1]["datetime"],
                "prediction": prediction,
                "actual_return": actual,
                "action": action,
                "realized_return": realized_return
            })

    # Build result DataFrame
    df_result = pd.DataFrame(records)
    df_result["cumulative_return"] = df_result["realized_return"].cumsum()
    df_result["real_cumulative_return"] = df_result["actual_return"].cumsum()

    return df_result



exit()

date_time = pd.to_datetime(validation_df.pop("datetime"), format="%Y-%m-%d")



# Create subplots: 2 rows (returns, predictions) × N models (columns)
fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 10), sharex=True)

# Make sure axes is always 2D
if len(models) == 1:
    axes = axes.reshape(2, 1)

# Iterate through models with index
for idx, (name, model) in enumerate(models.items()):
    result_df = simulate_trading(model, validation_df, input_width=180, target_col="COTTON", threshold=0.0)

    # Subplot for cumulative returns
    ax_returns = axes[0, idx]
    ax_returns.plot(result_df["cumulative_return"], label="Cumulative Return", linewidth=2)
    ax_returns.plot(result_df["real_cumulative_return"], label="Real Return", linewidth=2)
    ax_returns.set_title(f"{name} - Cumulative Return")
    ax_returns.set_xlabel("Date")
    ax_returns.set_ylabel("Log Return")
    ax_returns.grid(True)
    ax_returns.legend()

    # Subplot for predictions vs actual
    ax_forecast_vs_real = axes[1, idx]
    ax_forecast_vs_real.plot(validation_df["COTTON"], label="Actual", alpha=0.2)
    ax_forecast_vs_real.plot(result_df["prediction"], label="Prediction", alpha=0.7)
    ax_forecast_vs_real.set_title(f"{name} - Predictions vs Actual")
    ax_forecast_vs_real.set_xlabel("Date")
    ax_forecast_vs_real.set_ylabel("Price")
    ax_forecast_vs_real.grid(True)
    ax_forecast_vs_real.legend()

plt.tight_layout()
plt.show()
