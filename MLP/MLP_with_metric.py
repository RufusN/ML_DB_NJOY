import os
import glob
import re
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# Data loading functions
# ---------------------------
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
    return time_bins, frequencies, spectrogram_complex

def load_data_from_h5(directory):
    """
    Loads spectrograms (real & imaginary parts) from each .h5 file.
    Returns T_values, spectrogram data, file_list, and reference axes.
    """
    T_values = []
    real_imag_list = []
    file_list = []
    time_bins_ref, freq_ref = None, None
    T_min = 800
    T_max = 1200

    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    test_Ts = [1000]  # reserved for later inference

    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        match = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(file_path))
        temperature = float(match.group(1))
        if (T_min <= temperature <= T_max) and (temperature not in test_Ts):
            filename = os.path.basename(file_path)
            time_bins, frequencies, spectrogram_complex = read_spectrogram_h5(file_path)
            real_part = np.real(spectrogram_complex)
            imag_part = np.imag(spectrogram_complex)
            T_values.append(temperature)
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

# ---------------------------
# Model building function
# ---------------------------
def build_simple_model(input_shape, output_shape, hidden_units):
    """
    Builds a minimal model with one hidden layer of given size.
    The network maps a single scalar (temperature) to a spectrogram.
    """
    T_input = layers.Input(shape=input_shape, name='T_input')
    x = layers.Dense(hidden_units, activation='relu')(T_input)
    final_size = np.prod(output_shape)
    x = layers.Dense(final_size, activation='linear')(x)
    output = layers.Reshape(output_shape)(x)
    model = models.Model(inputs=T_input, outputs=output)
    return model

# ---------------------------
# Custom metric: maximum absolute error
# ---------------------------
def max_absolute_error_metric(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred)/tf.abs(y_true))

# ---------------------------
# Optimization script
# ---------------------------
def optimize_model():
    # 1. Load and preprocess data
    data_directory = '/Volumes/T7 Shield/T_800_1200_data/3x170_spectrograms'
    T, real_imag, file_list, time_bins_ref, freq_ref, test_Ts = load_data_from_h5(data_directory)
    print(f"Files processed: {len(file_list)}")
    print("Real+Imag shape:", real_imag.shape)
    
    N, h, w, c = real_imag.shape
    real_imag_flat = real_imag.reshape(N, -1)
    
    scaler_T = StandardScaler()
    T_norm = scaler_T.fit_transform(T)
    
    scaler_spec = StandardScaler()
    real_imag_scaled_flat = scaler_spec.fit_transform(real_imag_flat)
    real_imag_scaled = real_imag_scaled_flat.reshape((N, h, w, c))
    
    # 2. Split data: 80% train, 20% validation (for optimization)
    T_train, T_val, spec_train, spec_val = train_test_split(
        T_norm, real_imag_scaled, test_size=0.2, random_state=42
    )
    
    # 3. Define search over candidate hidden layer sizes.
    threshold = 8e-4
    best_candidate = None
    best_complexity = np.inf
    best_val_error = None
    best_units = None
    
    # We'll search over a range of hidden_units (e.g. 1 to 20)
    for hidden_units in range(1, 21):
        print(f"\nTraining model with hidden_units = {hidden_units}")
        model = build_simple_model(input_shape=(1,), output_shape=(h, w, c), hidden_units=hidden_units)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',  # Using MAE so that loss equals average absolute error
            metrics=[max_absolute_error_metric]
        )
        
        # Use early stopping to avoid overtraining
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        
        history = model.fit(
            T_train, spec_train,
            epochs=100,
            batch_size=32,
            validation_data=(T_val, spec_val),
            callbacks=[early_stop],
            verbose=0  # Set to 1 to see training details
        )
        
        # Evaluate on the validation set
        results = model.evaluate(T_val, spec_val, verbose=0)
        # results[0] is loss (MAE) and results[1] is our custom max absolute error metric
        val_max_error = results[1]
        complexity = model.count_params()
        print(f"Candidate with {hidden_units} units has {complexity} parameters and validation max abs error = {val_max_error:.6f}")
        
        if val_max_error < threshold:
            print(f"--> Candidate meets threshold (< {threshold}).")
            # We want the minimal complexity model that meets the threshold
            if complexity < best_complexity:
                best_candidate = model
                best_complexity = complexity
                best_val_error = val_max_error
                best_units = hidden_units
                # Since we're iterating from low to high, we could break here if we assume monotonicity.
                # However, if not, you can continue searching.
                # For this example, we break on the first candidate that meets the threshold.
                break
        else:
            print(f"--> Candidate does not meet threshold.")
    
    if best_candidate is not None:
        print(f"\nBest candidate: hidden_units = {best_units} with {best_complexity} parameters and validation max abs error = {best_val_error:.6f}")
    else:
        print("\nNo candidate model met the threshold. Consider expanding the search range or adjusting training hyperparameters.")
    
    # Optionally, you can evaluate the best candidate on a separate test set
    # (if you reserved one) and then proceed with inference.
    
    return best_candidate, scaler_T, scaler_spec, time_bins_ref, freq_ref, test_Ts

def inference(best_model, scaler_T, scaler_spec, time_bins_ref, freq_ref, test_Ts):
    # For each test temperature, predict and save the spectrogram
    for new_T in test_Ts:
        new_T_norm = scaler_T.transform([[new_T]])
        pred_scaled = best_model.predict(new_T_norm).squeeze()
        # Save scalers for later inverse-transform
        with h5py.File("spec_scalers.h5", "w") as hf:
            hf.create_dataset("spec_scale", data=scaler_spec.scale_)
            hf.create_dataset("spec_mean", data=scaler_spec.mean_)
            hf.create_dataset("T_scale", data=scaler_T.scale_)
            hf.create_dataset("T_mean", data=scaler_T.mean_)
        # Inverse scaling
        pred_scaled_flat = pred_scaled.reshape(1, -1)
        pred_unscaled_flat = scaler_spec.inverse_transform(pred_scaled_flat)
        # Reshape to spectrogram dimensions
        N, h, w, c = 1, len(time_bins_ref), len(freq_ref), 2
        pred_unscaled = pred_unscaled_flat.reshape(h, w, c)
        pred_complex = pred_unscaled[..., 0] + 1j * pred_unscaled[..., 1]
        output_filename = f"../ML_prediction/predicted_spectrogram_{new_T}_real_imag.h5"
        with h5py.File(output_filename, "w") as h5f:
            h5f.create_dataset("time_bins", data=time_bins_ref)
            h5f.create_dataset("frequencies", data=freq_ref)
            h5f.create_dataset("spectrogram_real", data=pred_unscaled[..., 0])
            h5f.create_dataset("spectrogram_imag", data=pred_unscaled[..., 1])
        print(f"Saved predicted spectrogram to {output_filename}")

if __name__ == "__main__":
    best_model, scaler_T, scaler_spec, time_bins_ref, freq_ref, test_Ts = optimize_model()
    if best_model is not None:
        inference(best_model, scaler_T, scaler_spec, time_bins_ref, freq_ref, test_Ts)
