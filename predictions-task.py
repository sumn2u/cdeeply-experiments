import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network on weather data.")

    # File input
    parser.add_argument('--file', type=str, required=True, help="Path to dataset (e.g. seoul_weather.txt)")

    # Use column indices for targets
    parser.add_argument('--mode', choices=['columns', 'rows'], default='columns',
                        help="Predict 'columns' (recommended) or 'rows'")
    parser.add_argument('--targets', type=str, required=True,
                        help="Comma-separated column indices to predict (e.g. 22,23)")

    # NN options
    parser.add_argument('--max_layers', type=int, default=3)
    parser.add_argument('--max_neurons', type=int, default=64)
    parser.add_argument('--max_weights', type=int, default=1000000)
    parser.add_argument('--has_bias', action='store_true')
    parser.add_argument('--sparse_weights', action='store_true')  # placeholder

    # Activation
    parser.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh', 'step', 'relu1'], default='tanh')

    # Quantization
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_zero', type=int, default=8)
    parser.add_argument('--quant_range', type=float, default=1.5)

    # Save predictions
    parser.add_argument('--save_csv', action='store_true', help="Save predictions to predictions_output.csv")

    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path, delimiter=',')
    df.dropna(inplace=True)
    print(f"✅ Loaded dataset: {path} with shape {df.shape}")
    return df


def build_model(input_shape, output_dim, layers, neurons, activation, has_bias):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(input_shape,), use_bias=has_bias))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation, use_bias=has_bias))
    model.add(Dense(output_dim))
    return model


def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_model = converter.convert()
    with open("model_quantized.tflite", "wb") as f:
        f.write(quant_model)
    print("📦 Model quantized and saved as model_quantized.tflite")


def evaluate_model(y_test, y_pred, y_scaler=None, target_names=None, save_csv=False):
    # De-normalize if scaler is available
    if y_scaler is not None:
        y_test = y_scaler.inverse_transform(y_test)
        y_pred = y_scaler.inverse_transform(y_pred)

    # Print metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid()
    plt.tight_layout()
    plt.savefig("prediction_vs_actual.png")
    print("📈 Saved plot: prediction_vs_actual.png")

    # Save CSV
    if save_csv:
        df_out = pd.DataFrame()
        if target_names:
            for i, name in enumerate(target_names):
                df_out[f"{name}_true"] = y_test[:, i]
                df_out[f"{name}_pred"] = y_pred[:, i]
        else:
            for i in range(y_test.shape[1]):
                df_out[f"y_true_{i}"] = y_test[:, i]
                df_out[f"y_pred_{i}"] = y_pred[:, i]
        df_out.to_csv("predictions_output.csv", index=False)
        print("💾 Saved predictions to predictions_output.csv")


def main():
    args = parse_arguments()
    data = load_data(args.file)

    # Print available columns
    print("\n📊 Available Columns:")
    for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

    target_indices = list(map(int, args.targets.split(',')))

    if args.mode == 'columns':
        try:
            X = data.drop(data.columns[target_indices], axis=1)
            y = data.iloc[:, target_indices]
            target_names = [data.columns[i] for i in target_indices]
            print(f"\n🎯 Predicting columns at indices: {target_indices}")
            print(f"   Column names: {target_names}")
        except IndexError as e:
            print(f"\n❌ Invalid target column indices: {e}")
            return
    else:
        X = data.iloc[:-1]
        y = data.iloc[1:]
        target_names = list(data.columns)

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_model(X.shape[1], y.shape[1], args.max_layers, args.max_neurons, args.activation, args.has_bias)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train
    print("\n🚀 Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, y_scaler=scaler_y, target_names=target_names, save_csv=args.save_csv)

    # Quantization
    if args.quantize:
        quantize_model(model)


if __name__ == '__main__':
    main()
