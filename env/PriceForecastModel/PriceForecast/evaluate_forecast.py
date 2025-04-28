import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.use('TkAgg')


def evaluate_forecast(model, X_test, y_test, scaler, save_folder=None, suffix=None, preffix=None):
    # Plot predictions vs. true values for visualization
    y_pred = model.predict(X_test)

    # Rescale predictions and true values back to original scale
    y_test_rescaled = scaler.inverse_transform(y_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))

    if save_folder is not None:
        if suffix is not None and type(suffix)==str:
            suffix = f" ({suffix})"
        else:
            suffix = ""

        if preffix is not None and type(preffix)==str:
            preffix = f"({preffix}) "
        else:
            preffix = ""

        for i in range(10):
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_rescaled[i * 100][:], label='True Prices')
            plt.plot(y_pred_rescaled[i * 100][:], label='Predicted Prices')
            plt.title(f'{preffix}Predicted vs. True Prices (Sample {i + 1}){suffix}')
            plt.xlabel('t_step')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(os.path.join(save_folder, f'{preffix}price_forecasting_sample_{i + 1}{suffix}.png'))
            plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.suptitle('Results Price Forecasting')
        ax1.set_title('Predicted vs. True Prices')
        ax1.set_xlabel('t_step')
        ax1.set_ylabel('Price')
        ax1.plot(y_test_rescaled[0][:], label='True Prices')
        ax1.plot(y_pred_rescaled[0][:], label='Predicted Prices')
        ax1.legend()
        ax2.set_title(f'{preffix}RMSE Loss{suffix}')
        ax2.set_xlabel('# Test Data')
        ax2.set_ylabel('RMSE')
        ax2.plot(rmse, label='RMSE Loss')
        ax2.legend()
        plt.show(block=False)
        plt.savefig(os.path.join(save_folder, f'{preffix}evaluation_price_forecasting_model{suffix}.png'))
        plt.close()

    return rmse

def evaluate_forecast_LSTM_Transformer(model, X_test, y_test, scaler, save_folder=None, suffix=None, preffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Plot predictions vs. true values for visualization
    print("test starting")
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float).to(device)  # 将输入张量移动到指定设备
    # print(X_test.shape)

    y_pred = []
    for i in range(len(X_test)):
        X_test_i = X_test[i]
        X_test_i = X_test_i.unsqueeze(0)
        X_test_i = X_test_i.unsqueeze(-1)     #(1,672,1)

        with torch.no_grad():
            y_pred_out = model(X_test_i)     #(1,48,1)
        y_pred.append(y_pred_out)

    y_pred = torch.cat(y_pred, dim=0)         #(6884, 48, 1)
    y_pred = y_pred.squeeze(-1)               #(6884, 48)
    y_pred = y_pred.cpu().numpy()  #


    # Rescale predictions and true values back to original scale
    y_test_rescaled = scaler.inverse_transform(y_test)
    # print(y_test_rescaled.shape)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    # print(y_pred_rescaled.shape)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))

    if save_folder is not None:
        if suffix is not None and type(suffix)==str:
            suffix = f" ({suffix})"
        else:
            suffix = ""

        if preffix is not None and type(preffix)==str:
            preffix = f"({preffix}) "
        else:
            preffix = ""

        for i in range(10):
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_rescaled[i * 100][:], label='True Prices')
            plt.plot(y_pred_rescaled[i * 100][:], label='Predicted Prices')
            plt.title(f'{preffix}Predicted vs. True Prices (Sample {i + 1}){suffix}')
            plt.xlabel('t_step')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(os.path.join(save_folder, f'{preffix}price_forecasting_sample_{i + 1}{suffix}.png'))
            plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.suptitle('Results Price Forecasting')
        ax1.set_title('Predicted vs. True Prices')
        ax1.set_xlabel('t_step')
        ax1.set_ylabel('Price')
        ax1.plot(y_test_rescaled[0][:], label='True Prices')
        ax1.plot(y_pred_rescaled[0][:], label='Predicted Prices')
        ax1.legend()
        ax2.set_title(f'{preffix}RMSE Loss{suffix}')
        ax2.set_xlabel('# Test Data')
        ax2.set_ylabel('RMSE')
        ax2.plot(rmse, label='RMSE Loss')
        ax2.legend()
        plt.show(block=False)
        plt.savefig(os.path.join(save_folder, f'{preffix}evaluation_price_forecasting_model{suffix}.png'))
        plt.close()

    return rmse

def evaluate_forecast_MD(model, X_test, y_test, save_folder=None, suffix=None, preffix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Plot predictions vs. true values for visualization
    print("test starting")
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    # print(X_test.shape)

    y_pred = []
    for i in range(len(X_test)):
        # print(f"第{i}组:")
        X_test_i = X_test[i]
        X_test_i = X_test_i.unsqueeze(0)
        X_test_i = X_test_i.unsqueeze(-1)     #(1,672,1)
        # print(X_test_i.shape)
        with torch.no_grad():
            # print(X_test_i)
            y_pred_out = model(X_test_i)     #(1,48,1)
            # print(y_pred_out)
        y_pred.append(y_pred_out)

    y_pred = torch.cat(y_pred, dim=0)         #(6884, 48, 1)
    y_pred = y_pred.squeeze(-1)               #(6884, 48)
    y_pred = y_pred.cpu().numpy()
    # print(y_pred.shape)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=1))

    if save_folder is not None:
        if suffix is not None and type(suffix)==str:
            suffix = f" ({suffix})"
        else:
            suffix = ""

        if preffix is not None and type(preffix)==str:
            preffix = f"({preffix}) "
        else:
            preffix = ""

        for i in range(10):
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[i * 100][:], label='True Prices')
            plt.plot(y_pred[i * 100][:], label='Predicted Prices')
            plt.title(f'{preffix}Predicted vs. True Prices (Sample {i + 1}){suffix}')
            plt.xlabel('t_step')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(os.path.join(save_folder, f'{preffix}price_forecasting_sample_{i + 1}{suffix}.png'))
            plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.suptitle('Results Price Forecasting')
        ax1.set_title('Predicted vs. True Prices')
        ax1.set_xlabel('t_step')
        ax1.set_ylabel('Price')
        ax1.plot(y_test[0][:], label='True Prices')
        ax1.plot(y_pred[0][:], label='Predicted Prices')
        ax1.legend()
        ax2.set_title(f'{preffix}RMSE Loss{suffix}')
        ax2.set_xlabel('# Test Data')
        ax2.set_ylabel('RMSE')
        ax2.plot(rmse, label='RMSE Loss')
        ax2.legend()
        plt.show(block=False)
        plt.savefig(os.path.join(save_folder, f'{preffix}evaluation_price_forecasting_model{suffix}.png'))
        plt.close()

    return rmse