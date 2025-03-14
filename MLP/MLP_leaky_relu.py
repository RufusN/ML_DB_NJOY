import os
import glob
import re
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LeakyReLU
import math

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
    Loads the real and imaginary parts of the spectrogram from each .h5 file.
    Expects shape (10, 1940).
    Returns T_values, real+imag arrays, file_list, and reference axes.
    """
    T_values = []
    real_imag_list = []
    file_list = []
    time_bins_ref, freq_ref = None, None
    T_min = 950
    T_max = 1050
    test_plit = 0.0

    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    test_Ts = []
    test_Ts = [1000]

    # for file_path in all_files:
    #     file_path = os.path.normpath(file_path)
    #     temperature = float(os.path.basename(file_path).split('_')[-1].split('.')[0])
    #     if (((temperature >= T_min) and (temperature <= T_max))):
    #         r = np.random.rand()
    #         if (r < test_plit):
    #             test_Ts.append(temperature)

    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        # temperature = float(os.path.basename(file_path).split('_')[-1].split('.')[0])
        # Use regex to extract the temperature value
        match = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(file_path))
        temperature = float(match.group(1))
        if (((temperature >= T_min) and (temperature <= T_max)) and (temperature not in test_Ts) ): #and (round(math.modf(temperature)[0], 10) in {0.0, 0.2, 0.4, 0.6, 0.8})
            # print(temperature, round(math.modf(temperature)[0], 10) in {0.0, 0.2, 0.4, 0.6, 0.8} )
            filename = os.path.basename(file_path)
            time_bins, frequencies, spectrogram_complex = read_spectrogram_h5(file_path)

            # Extract real and imaginary parts
            real_part = np.real(spectrogram_complex)
            imag_part = np.imag(spectrogram_complex)

            T_values.append(temperature)
            # Combine along a new last dimension => shape (10, 1940, 2)
            real_imag = np.stack([real_part, imag_part], axis=-1)
            real_imag_list.append(real_imag)
            file_list.append(filename)
            print(f"Loaded {file_path}")

            # Save reference axes if not set
            if time_bins_ref is None and freq_ref is None:
                time_bins_ref = time_bins
                freq_ref = frequencies

    T_values = np.array(T_values, dtype=np.float32).reshape(-1, 1)
    real_imag_data = np.array(real_imag_list, dtype=np.float32)  # shape (N, 10, 1940, 2)
    return T_values, real_imag_data, file_list, time_bins_ref, freq_ref, test_Ts

def build_real_imag_model(input_shape, output_shape, l2_reg=0.0, dropout_rate=0.0, leaky_relu_alpha=0.1):
    """
    Builds a regularized model with Dropout, L2 regularization, and Leaky ReLU activations.
    
    Parameters:
    - input_shape: Tuple, shape of the input (e.g., (1,))
    - output_shape: Tuple, desired output shape 
    - l2_reg: Float, L2 regularization factor
    - dropout_rate: Float, Dropout rate between 0 and 1
    - leaky_relu_alpha: Float, negative slope coefficient for Leaky ReLU
    """
    T_input = layers.Input(shape=input_shape, name='T_input')
    
    x = layers.Dense(16, 
                     kernel_regularizer=regularizers.l2(l2_reg),
                     use_bias=True)(T_input)
    # Leaky ReLU activation
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    # Dropout
    # x = layers.Dropout(dropout_rate)(x)
    
    # x = layers.Dense(16, 
    #                 kernel_regularizer=regularizers.l2(l2_reg),
    #                 use_bias=True)(T_input)
    # # Leaky ReLU activation
    # x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    # Dropout
    # x = layers.Dropout(dropout_rate)(x)
    
    # Output Dense layer without activation
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
    # data_directory = '/Volumes/T7 Shield/T_800_1200_data/107x709_spectrograms'
    data_directory = '/Volumes/T7 Shield/T_800_1200_data/test_2'
    # data_directory = '/Volumes/T7 Shield/T_800_1200_data/54x709_spectrograms'
    # data_directory = '/Volumes/T7 Shield/T_800_1200_data/213x227_spectrograms'

    T, real_imag, file_list, time_bins_ref, freq_ref, test_Ts = load_data_from_h5(data_directory)
    print(f"Files processed: {len(file_list)}")
    print("Real+Imag shape:", real_imag.shape)

    N, h, w, c = real_imag.shape 
    num_pixels = h * w * c 
    real_imag_flat = real_imag.reshape(N, num_pixels)
    spec_mean, spec_std = np.mean(real_imag_flat), np.std(real_imag_flat, ddof=0)
    T_mean, T_std = np.mean(T), np.std(T, ddof=0)

    with open("spectrogram_stats.txt", "w") as f:
        f.write(f"Spectrogram values: mean={spec_mean}, std={spec_std}\n")
        f.write(f"T values: mean={T_mean}, std={T_std}\n")


    # Scale T to [0,1]
    scaler_T = StandardScaler()
    T_norm = scaler_T.fit_transform(T)

    # Scale real+imag across entire dataset
    scaler_spec = StandardScaler() 
    real_imag_scaled_flat = (scaler_spec.fit_transform(real_imag_flat)) #np.log
    real_imag_scaled = real_imag_scaled_flat.reshape((N, h, w, c))

    # -----------------------
    # 4. Train/test split
    # -----------------------
    T_train, T_test, spec_train, spec_test = train_test_split(
        T_norm, real_imag_scaled, test_size=0.2, random_state=42
    )

    # -----------------------
    # 5. Build model
    # -----------------------
    # model = build_real_imag_model(input_shape=(1,), output_shape=(h, w, c))
    model = build_real_imag_model(input_shape=(1,), output_shape=(h, w, c))
    model.summary()

    # -----------------------
    # 6. Compile
    #    We'll use mse on the scaled (real,imag) domain
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
    checkpoint = ModelCheckpoint('best_model_real_imag.keras', monitor='val_loss', save_best_only=True, verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # -----------------------
    # 8. Train
    # -----------------------
    history = model.fit(
        T_train, spec_train,
        epochs=60,
        batch_size=32,
        validation_split=0.2,
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

        new_T_norm = scaler_T.transform([[new_T]])  
        pred_scaled = model.predict(new_T_norm) 
        pred_scaled = pred_scaled.squeeze()  
        print("Prescaled output from the MLP:", pred_scaled[:3, 83])    
        # print(scaler_spec.scale_)
        print(pred_scaled.shape)
        print(np.shape(scaler_spec.mean_))

        # Ensure the whole array is printed.
        np.set_printoptions(threshold=np.inf)

        with open("scaler_stats.txt", "w") as f:
            f.write("Scaler Spec Scale:\n")
            f.write(np.array2string(scaler_spec.scale_, separator=', '))
            f.write("\n\nScaler Spec Mean:\n")
            f.write(np.array2string(scaler_spec.mean_, separator=', '))



        # Inverse scale
        pred_scaled_flat = (pred_scaled.reshape(1, -1)) #np.exp
        pred_unscaled_flat = scaler_spec.inverse_transform(pred_scaled_flat)
        pred_unscaled = pred_unscaled_flat.reshape(h, w, c)  # shape (10,1940,2)

        # Now pred_unscaled[...,0] = real part, pred_unscaled[...,1] = imaginary part
        # Combine into a complex spectrogram:
        pred_complex = pred_unscaled[...,0] + 1j * pred_unscaled[...,1]

        # -----------------------
        # 11. Save results to HDF5
        # -----------------------
        output_filename = f"../../data/ML_preds/predicted_spectrogram_{new_T}_real_imag.h5"
        with h5py.File(output_filename, "w") as h5f:
            h5f.create_dataset("time_bins", data=time_bins_ref)
            h5f.create_dataset("frequencies", data=freq_ref)
            h5f.create_dataset("spectrogram_real", data=pred_unscaled[...,0])
            h5f.create_dataset("spectrogram_imag", data=pred_unscaled[...,1])
        print(f"Saved predicted real+imag spectrogram + axes to {output_filename}")

if __name__ == "__main__":
    main()
