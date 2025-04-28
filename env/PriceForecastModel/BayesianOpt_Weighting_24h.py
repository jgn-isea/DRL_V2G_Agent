import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
import PriceForecast as pf
import torch
import numpy as np

# Step 1: Load and preprocess data
data = pd.read_csv('./LSTM/data/electricity_prices_ID1.csv')
input_length = 168 * 4
output_length = 24 * 4
X, y = pf.create_sequences(data, input_length=input_length, output_length=output_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training data
X_train, X_test, y_train, y_test = pf.preprocess_data(X_train, X_test, y_train, y_test, q_lb=0.05, q_ub=0.95)

timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
save_folder = os.path.join("./LSTM/PriceForecast_md_res/","BayesianOpt_Weighting_24h_"+ timestamp )
os.makedirs(save_folder, exist_ok=True)

trial_results = []

# Define the objective function for Bayesian optimization
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.003, log=True)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 5)

    lr_decay_factor = trial.suggest_float('lr_decay_factor', 0.8, 0.9, log=True)
    batch_size = trial.suggest_categorical('batch_size', [48, 64, 80])
    lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.001, 0.05, log=True)

    trans_heads = 4
    activation = 'relu'
    trans_layers = 2
    trans_ffn_hidden = 64

    # Build and train the model
    model, train_losses = pf.build_and_train_Transformer_bys_weighting(
        X_train, y_train,
        batch_size=batch_size,
        output_window=output_length,
        epochs=100,
        input_size=1,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        trans_layers=trans_layers,
        trans_heads=trans_heads,
        trans_ffn_hidden=trans_ffn_hidden,
        dropout=dropout,
        output_size=1,
        learning_rate=learning_rate,
        lr_decay_factor=lr_decay_factor,
        activation=activation
    )

    final_loss = train_losses[-1]

    trial_info = {
        'trial_number': trial.number,
        'parameters': trial.params,
        'final_loss': final_loss
    }
    trial_results.append(trial_info)

    return final_loss

# Create a study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Output the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

best_params_path = os.path.join(save_folder, 'best_hyperparameters.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params, f, indent=4)

trial_results_path = os.path.join(save_folder, 'trial_results.json')
with open(trial_results_path, 'w') as f:
    json.dump(trial_results, f, indent=4)

# Train a model with the best hyperparameters
num_epochs = 800
best_model, train_losses = pf.build_and_train_Transformer_bys_weighting(
    X_train, y_train,
    batch_size=best_params['batch_size'],
    output_window=output_length,
    epochs=num_epochs,  
    input_size=1,
    lstm_hidden_size=best_params['lstm_hidden_size'],
    lstm_layers=best_params['lstm_layers'],
    trans_layers=2,
    trans_heads=4,
    trans_ffn_hidden=64,
    dropout=best_params['dropout'],
    output_size=1,
    learning_rate=best_params['learning_rate'],
    lr_decay_factor=best_params['lr_decay_factor'],
    activation='relu'
)

eval_folder = os.path.join(save_folder, f"Train_Evaluate_{num_epochs}s")
os.makedirs(eval_folder, exist_ok=True)
model_path = os.path.join(eval_folder, f'lstm_transformer_bys_best_weighting_{num_epochs}s.pth')
torch.save(best_model, model_path)

rmse = pf.evaluate_forecast_MD(best_model, X_test, y_test, save_folder=eval_folder)

# Visualization of optimization results
import optuna.visualization as vis

# Plot optimization history
fig_history = vis.plot_optimization_history(study)
fig_history.show()

# Plot hyperparameter importance
fig_importances = vis.plot_param_importances(study)
fig_importances.show()

# Save visualizations
fig_history.write_html(os.path.join(save_folder, 'optimization_history.html'))
fig_importances.write_html(os.path.join(save_folder, 'param_importances.html'))