import os
import glob
import re
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
import math
import time
import keras_tuner as kt  # Alternative hyperparameter tuning framework

# -----------------------
# Data loading functions
# -----------------------
def read_spectrogram_h5(h5_filepath):
    with h5py.File(h5_filepath, 'r') as h5f:
        required_datasets = {'time_bins', 'frequencies', 'spectrogram_real', 'spectrogram_imag'}
        missing = required_datasets - set(h5f.keys())
        if missing:
            raise KeyError(f"Missing datasets in HDF5 file: {missing}")
        time_bins = h5f['time_bins'][:]
        frequencies = h5f['frequencies'][:]
        spectrogram_real = h5f['spectrogram_real'][:]
        spectrogram_imag = h5f['spectrogram_imag'][:]
    spectrogram_complex = spectrogram_real + 1j * spectrogram_imag
    print(np.shape(spectrogram_complex))
    return time_bins, frequencies, spectrogram_complex

def load_data_from_h5(directory):
    """
    Loads spectrogram data from HDF5 files.
    Returns T_values, real+imag data, list of files, and reference axes.
    """
    T_values = []
    real_imag_list = []
    file_list = []
    time_bins_ref, freq_ref = None, None
    T_min = 800
    T_max = 1200
    test_Ts = [1000]  # fixed test temperature

    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        match = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(file_path))
        if match is None:
            continue
        temperature = float(match.group(1))
        if (T_min <= temperature <= T_max) and (temperature not in test_Ts):
            filename = os.path.basename(file_path)
            time_bins, frequencies, spectrogram_complex = read_spectrogram_h5(file_path)
            real_part = np.real(spectrogram_complex)
            imag_part = np.imag(spectrogram_complex)
            T_values.append(temperature)
            # Combine real and imaginary parts along a new last dimension: shape (10, 1940, 2)
            real_imag = np.stack([real_part, imag_part], axis=-1)
            real_imag_list.append(real_imag)
            file_list.append(filename)
            print(f"Loaded {file_path}")
            if time_bins_ref is None and freq_ref is None:
                time_bins_ref = time_bins
                freq_ref = frequencies

    T_values = np.array(T_values, dtype=np.float32).reshape(-1, 1)
    real_imag_data = np.array(real_imag_list, dtype=np.float32)
    return T_values, real_imag_data, file_list, time_bins_ref, freq_ref, test_Ts

def log_ae_loss(y_true, y_pred, epsilon=1e-16):
    return tf.reduce_mean(tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon))

# -----------------------
# Custom callback to measure inference time and compute a composite metric
# -----------------------
class InferenceTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_input, lambda_weight=0.01):
        """
        sample_input: a small batch from validation data on which inference time is measured.
        lambda_weight: weight to balance the contribution of inference time in the combined metric.
        """
        super(InferenceTimeCallback, self).__init__()
        self.sample_input = sample_input
        self.lambda_weight = lambda_weight

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        start_time = time.perf_counter()
        _ = self.model.predict(self.sample_input, verbose=0)
        inference_time = time.perf_counter() - start_time
        avg_inference_time = inference_time / self.sample_input.shape[0]
        # Add inference time and combined metric to logs
        logs['inference_time'] = avg_inference_time
        logs['val_combined'] = logs.get('val_loss', 0) + self.lambda_weight * avg_inference_time
        print(f"Epoch {epoch+1}: Inference time = {avg_inference_time:.6f} sec, Combined metric = {logs['val_combined']:.6f}")

# -----------------------
# Main: Data loading, model tuning, and inference time measurement
# -----------------------
def main():
    # Change this to your data directory
    data_directory = '/Volumes/T7 Shield/T_800_1200_data/3x170_spectrograms'
    
    # Load data from HDF5 files
    T, real_imag, file_list, time_bins_ref, freq_ref, test_Ts = load_data_from_h5(data_directory)
    print(f"Files processed: {len(file_list)}")
    print("Real+Imag shape:", real_imag.shape)

    N, h, w, c = real_imag.shape
    num_pixels = h * w * c
    real_imag_flat = real_imag.reshape(N, num_pixels)

    # Scale T
    scaler_T = StandardScaler()
    T_norm = scaler_T.fit_transform(T)

    # Scale real+imag data
    scaler_spec = StandardScaler() 
    real_imag_scaled_flat = scaler_spec.fit_transform(real_imag_flat)
    real_imag_scaled = real_imag_scaled_flat.reshape((N, h, w, c))

    # Split into training and validation sets
    T_train, T_val, spec_train, spec_val = train_test_split(
        T_norm, real_imag_scaled, test_size=0.2, random_state=42
    )

    # Define output shape as a global variable for the model builder
    output_shape = (h, w, c)

    # -----------------------
    # Define the model builder function for KerasTuner
    # -----------------------
    def build_model(hp):
        # Hyperparameters to tune
        hidden_units = hp.Int('hidden_units', min_value=1, max_value=12, step=1)
        l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log')
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5)
        leaky_relu_alpha = hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3)
        lr = hp.Float('lr', min_value=1e-4, max_value=1e-1, sampling='log')
        
        # Build the MLP model
        T_input = layers.Input(shape=(1,), name='T_input')
        x = layers.Dense(hidden_units,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         use_bias=True)(T_input)
        x = LeakyReLU(alpha=leaky_relu_alpha)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        
        final_size = np.prod(output_shape)
        x = layers.Dense(final_size,
                         kernel_regularizer=regularizers.l2(l2_reg),
                         activation='linear',
                         use_bias=True)(x)
        output = layers.Reshape(output_shape)(x)
        model = models.Model(inputs=T_input, outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=log_ae_loss,
            metrics=['mse', 'mae']
        )
        return model

    # -----------------------
    # Create a KerasTuner Hyperband tuner using the composite metric
    # -----------------------
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_combined", direction="min"),
        max_epochs=30,
        factor=3,
        directory='keras_tuner_dir',
        project_name='mlp_inference_tuning'
    )

    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # Prepare a small sample from the validation set for inference time measurement
    sample_input_for_callback = T_val[:10]
    inference_time_callback = InferenceTimeCallback(sample_input=sample_input_for_callback, lambda_weight=0.01)

    # Run the search with the custom callback
    tuner.search(T_train, spec_train,
                 epochs=30,
                 validation_data=(T_val, spec_val),
                 callbacks=[early_stop, inference_time_callback],
                 verbose=1)

    # Retrieve the best hyperparameters and best model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    print("\nBest Hyperparameters:")
    for key, value in best_hp.values.items():
        print(f"  {key}: {value}")

    # -----------------------
    # Measure inference time of the best model (final evaluation)
    # -----------------------
    sample_input = T_val[:10]
    start_time = time.perf_counter()
    _ = best_model.predict(sample_input, verbose=0)
    inference_time = time.perf_counter() - start_time
    avg_inference_time = inference_time / sample_input.shape[0]
    print(f"\nAverage inference time per sample: {avg_inference_time:.6f} seconds")

    # Optionally: Evaluate the best model on validation data
    val_results = best_model.evaluate(T_val, spec_val, verbose=2)
    print("Validation results (loss, mse, mae):", val_results)

if __name__ == "__main__":
    main()
