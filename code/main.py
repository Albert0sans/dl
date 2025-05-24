from utils.window_generator import WindowGenerator
from utils.preproces import trainTestSplit, preprocesDf,computeFeatures
#import utils.dl_training as DL
from utils.layers.transformer import transformer_timeseries
import utils.multi_class_training as mc
import utils.dl_layers as CustomLayers
from  utils.dl_layers import AutoregressiveWrapperLSTM,GenerativeAdversialEncoderWrapper
from utils.plotting import multiModelComparison
from utils.layers.informer.informer import Informer
import matplotlib.pyplot as plt
from utils.algos import simple_backtest
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns



OUT_STEPS = 1
INPUT_WIDTH=180
MAX_EPOCHS = 100
BATCH_SIZE=32
df = pd.read_csv("commodity_futures.csv")
df=df.sample(frac=1).sort_index()
# Shuffle the DataFrame

# Split into 80/20
split_idx = int(len(df) * 0.8)
validation_df = df[split_idx:]

df = df[:split_idx]

df = preprocesDf(df)
print(df)
df=computeFeatures(df)
n=len(df)
print(df)
#target_cols = ["TARGET_close_AAPL"]
target_cols=["COTTON"]
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
    head_size=32,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[32],
    mlp_dropout=0.2,
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
if False:
    informer.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mae, metrics=['mse', 'mae'],
                            run_eagerly=True)
    informer.fit(train_ds,
                        steps_per_epoch=20,
                        
                        validation_steps=20,
                        
                        epochs=10)


models = {
  #  "test":CustomLayers.ZeroBaseline(OUT_STEPS,num_features),
  #  "informer":informer,
   "transformer":transformer,
  "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
  #  "gan":GenerativeAdversialEncoderWrapper(OUT_STEPS=OUT_STEPS,generator=generator,discriminator=discriminator, num_features=in_features,),
   #"ar_lstmstatefull_model":AutoregressiveWrapperLSTM(OUT_STEPS= OUT_STEPS,num_features=in_features),
    "auto_encoder":CustomLayers.auto_encoder(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
 #   #"transformer_model":CustomLayers.random_forest(OUT_STEPS=OUT_STEPS, num_features=num_features),
   
 
  "rnn_model": CustomLayers.rnn_model(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
   "rnn_model_gru": CustomLayers.rnn_model_gru(INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
    "multi_dense_model":CustomLayers.multi_dense_model( INPUT_WIDTH=INPUT_WIDTH,OUT_STEPS=OUT_STEPS,in_features=in_features, out_features=num_features),
   #"extrarfsklearn":CustomLayers.extrarf(OUT_STEPS,num_features),
   
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
    model=mc.MultiClassModel(
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
                             )
   
    models[name] = model.fit() 
    metrics = model.evaluate()
    model_metrics[name]=metrics

# Plot all fitted models

#multiModelComparison(list(model_metrics.values()), list(models.keys()))


#multi_window.plot(list(models.values()), plot_col=target_cols[0])




fig, axes = plt.subplots(2, len(models),  )


for idx, key in enumerate(model_metrics):
    model = models[key]
    forecast = model.predict(X_test).flatten()

    true = y_test

    ax_forecast = axes[0, idx]
    ax_kde = axes[1, idx]
    ax_forecast.plot(forecast.flatten(), label="Forecast")
    ax_forecast.plot(true.flatten(), label="Real")
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
    simple_backtest(df, real_column="real", forecast_column="forecast")


plt.show()

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
           r=0
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
            print(prediction)
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


date_time = pd.to_datetime(validation_df.pop("datetime"), format="%Y-%m-%d")
result_df = simulate_trading(model, validation_df, input_width=180, target_col="COTTON", threshold=0.0)


# Plot
plt.figure(figsize=(12, 5))
plt.plot( result_df["cumulative_return"], label="Cumulative Return", linewidth=2)
plt.plot( result_df["real_cumulative_return"], label="real Return", linewidth=2)

plt.title("Model Cumulative Return Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.plot(result_df["prediction"])
plt.show()