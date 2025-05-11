import os
import glob
import re
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LeakyReLU
import math
from tensorflow.keras.activations import softplus

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
    T_values = []
    real_imag_list = []
    file_list = []
    time_bins_ref, freq_ref = None, None
    T_min = 950.0
    T_max = 1050.0

    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    test_Ts = []
    test_Ts = [1000.0]

    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        match = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(file_path))
        temperature = float(match.group(1))
        if (((temperature >= T_min) and (temperature <= T_max)) and (temperature not in test_Ts) ):
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

    # T_values = np.array(T_values, dtype=np.float32).reshape(-1, 1)
    # real_imag_data = np.array(real_imag_list, dtype=np.float32)
    T_values_fin = np.array(T_values, dtype=np.float32).reshape(-1, 1)
    real_imag_data_fin = np.array(real_imag_list, dtype=np.float32)

    T_values = []
    real_imag_data = []

    print(np.shape(T_values_fin), np.shape(real_imag_data_fin))
    for i,j in zip(T_values_fin, real_imag_data_fin):
        T_values.append(i)
        real_imag_data.append(j)

    for i,j in zip(T_values_fin, real_imag_data_fin):
       T_values.append(i )#+ np.random.normal(loc=0, scale=0.001, size=1))
       real_imag_data.append(j + np.abs(j)*np.random.normal(loc=0, scale=0.000001, size=1))
    
    for i,j in zip(T_values_fin, real_imag_data_fin):
       T_values.append(i )# + np.random.normal(loc=0, scale=0.001, size=1))
       real_imag_data.append(j + np.abs(j)*np.random.normal(loc=0, scale=0.000001, size=1))
    
    for i,j in zip(T_values_fin, real_imag_data_fin):
       T_values.append(i )# + np.random.normal(loc=0, scale=0.001, size=1))
       real_imag_data.append(j + np.abs(j)*np.random.normal(loc=0, scale=0.000001, size=1))

    T_values = np.array(T_values)
    real_imag_data = np.array(real_imag_data)
    return T_values, real_imag_data, file_list, time_bins_ref, freq_ref, test_Ts, T_min, T_max

def build_real_imag_model(input_shape, output_shape, l2_reg=0, dropout_rate=0, leaky_relu_alpha=0.1
):
    T_input = layers.Input(shape=input_shape, name='T_input')
    
    x = layers.Dense(16, 
                     kernel_regularizer=regularizers.l2(l2_reg),
                     use_bias=True)(T_input)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)    

    final_size = np.prod(output_shape)
    x = layers.Dense(final_size, 
                     kernel_regularizer=regularizers.l2(l2_reg),
                     activation='linear',
                     use_bias=True)(x)
    
    # Reshape to desired output shape
    output = layers.Reshape(output_shape)(x)
    
    model = models.Model(inputs=T_input, outputs=output)
    return model

def log_ae_loss(y_true, y_pred, epsilon=1e-16):
    return tf.reduce_mean((tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon)))

def main():
    # -----------------------
    # 1. Load data
    # -----------------------
    data_directory = '/mnt/d/new_specs_3_5313'

    T, real_imag, file_list, time_bins_ref, freq_ref, test_Ts, T_min, T_max = load_data_from_h5(data_directory)
    print(f"Files processed: {len(file_list)}")
    print("Real+Imag shape:", real_imag.shape)

    N, h, w, c = real_imag.shape 
    num_pixels = h * w * c 
    real_imag_flat = real_imag.reshape(N, num_pixels)

    scaler_T = StandardScaler()
    T_norm = scaler_T.fit_transform(T)

    scaler_spec = StandardScaler() 
    real_imag_scaled_flat = (scaler_spec.fit_transform(real_imag_flat))
    real_imag_scaled = real_imag_scaled_flat.reshape((N, h, w, c))

    # -----------------------
    # 4. Train/test split
    # -----------------------
    T_train, T_test, spec_train, spec_test = train_test_split(
        T_norm, real_imag_scaled, test_size=0.00001, random_state=42
    )

    # -----------------------
    # 5. Build model
    # -----------------------

    model = build_real_imag_model(input_shape=(1,), output_shape=(h, w, c))
    model.summary()

    # -----------------------
    # 6. Compile
    # -----------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss=log_ae_loss,
        metrics=['mse', 'mae']
    )

    # -----------------------
    # 7. Callbacks
    # -----------------------
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(
        f'best_model_3_5313.h5',   # use .h5
        monitor='val_loss',
        save_best_only=True,
        save_format='h5',               # explicitly request HDF5
        verbose=1
    )
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # -----------------------
    # 8. Train
    # -----------------------
    history = model.fit(
        T_train, spec_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint, lr_reduce],
        verbose=1
    )

    # Plot training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.savefig("./loss_history.png")
    plt.show()

    # -----------------------
    # 9. Evaluate
    # -----------------------
    test_results = model.evaluate(T_test, spec_test, verbose=2)
    print("Test results (MSE on real+imag scaled):", test_results)

    # -----------------------
    # 10. Inference
    # -----------------------
    for new_T in test_Ts:

        new_T_norm = scaler_T.transform([[new_T**3]])  
        # new_T_norm = np.array([[(new_T)**2]])  # shape (1, 1)
        pred_scaled = model.predict(new_T_norm) 
        pred_scaled = pred_scaled.squeeze()   

        with h5py.File(f"spec_scalers_3_5313.h5", "w") as hf:
            hf.create_dataset("spec_scale", data=scaler_spec.scale_)
            hf.create_dataset("spec_mean", data=scaler_spec.mean_)
            hf.create_dataset("T_scale", data=scaler_T.scale_)
            hf.create_dataset("T_mean", data=scaler_T.mean_)

        # Inverse scale
        pred_scaled_flat = (pred_scaled.reshape(1, -1)) 
        pred_unscaled_flat = scaler_spec.inverse_transform(pred_scaled_flat)
        pred_unscaled = pred_unscaled_flat.reshape(h, w, c) 

        pred_complex = pred_unscaled[...,0] + 1j * pred_unscaled[...,1]

        # -----------------------
        # 11. Save results to HDF5
        # -----------------------
        output_filename = f"/mnt/c/Users/marti/Documents/Martin/SCONE/code_stuff/Scripts/DB_sliced/ML_DB_sliced_pipeline/3x45551_point_recon/martin_pipeline/ML_predictions/predicted_spectrogram_{new_T}_real_imag.h5"
        with h5py.File(output_filename, "w") as h5f:
            h5f.create_dataset("time_bins", data=time_bins_ref)
            h5f.create_dataset("frequencies", data=freq_ref)
            h5f.create_dataset("spectrogram_real", data=pred_unscaled[...,0])
            h5f.create_dataset("spectrogram_imag", data=pred_unscaled[...,1])
        print(f"Saved predicted real+imag spectrogram + axes to {output_filename}")

if __name__ == "__main__":
    main()
