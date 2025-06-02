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
from sklearn.preprocessing import PolynomialFeatures

#burn-in på 50-100K så trenger dette for endepunkter.
# case 1: train on 300-800, test 400-700. 64 nodes v good, 50 nodes ok, either 2 noise + "0", "2", "5", "7" OR 3 noise + "0", "5". TEST 32 --doesnt work but 50 barely! But maybe better with 64 -> much better accuracy so can do lossy for memory and inference might not be affected. with fermiac / HPC can do 0.1K maybe better and can reduce nodes.
# case 2: train on 700-1800, test 800-1700. 64 nodes ok, 50 nodes ok. 3 noise and 1K. with fermiac / HPC can do 0.1K maybe better and can reduce nodes.

#or split T into 3 for 16 node. but need 0.1K. see results, take it from there.

#DO PARAMETRIC STUDY OF ABOVE ON FERMIAC / HPC FINAL BIT.

#more data seems to help for lower Temps? error in data defo a problem for training
#higher temps need less noise.

#AHA broken data hidden! more sensitive to broken data when more "real data" and less noise added, thats
# why better performance when add noise, compared to no noise!!

#45551 or #5313
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
    #T_min = 300.0
    #T_max = 800.0
    #T_min = 700.0
    #T_max = 1800.0
    T_min = 300.0
    T_max = 800.0

    all_files = glob.glob(os.path.join(directory, "spectrogram_T_*.h5"))
    #test_Ts = [400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0] 
    #test_Ts = [800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0]
    test_Ts = [350.0, 400.0, 420.0, 450.0, 480.0, 550.0, 600.0, 650.0, 700.0, 750.0]
    for file_path in all_files:
        file_path = os.path.normpath(file_path)
        match = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(file_path))
        if not match:
            continue

        temperature = float(match.group(1))


        #if (not temperature.is_integer()):
        #    continue

        if not (str(temperature)[-1:] in ["0", "4", "7"]):
            continue
        # now do your usual range + test-split exclusion
        if T_min <= temperature <= T_max and (temperature not in test_Ts):
            filename = os.path.basename(file_path)
            time_bins, frequencies, spectrogram_complex = read_spectrogram_h5(file_path)

            real_part = np.real(spectrogram_complex)
            imag_part = np.imag(spectrogram_complex)

            T_values.append(temperature)
            real_imag = np.stack([real_part, imag_part], axis=-1)
            real_imag_list.append(real_imag)
            file_list.append(filename)
            #print(f"Loaded {file_path}")
            print("-----", temperature)

            if time_bins_ref is None and freq_ref is None:
                time_bins_ref = time_bins
                freq_ref = frequencies

    T_values_fin = np.array(T_values, dtype=np.float32).reshape(-1, 1)
    real_imag_data_fin = np.array(real_imag_list, dtype=np.float32)

    T_values = []
    real_imag_data = []

    #optimal is 1 normal + 3 noise

    for i,j in zip(T_values_fin, real_imag_data_fin):
        T_values.append(i)
        real_imag_data.append(j)

    for i,j in zip(T_values_fin, real_imag_data_fin):
        T_values.append(i)
        real_imag_data.append(j + (np.abs(j))*np.random.normal(loc=0, scale=0.000001, size=1)) #0.00001
#
    for i,j in zip(T_values_fin, real_imag_data_fin):
        T_values.append(i)
        real_imag_data.append(j + (np.abs(j))*np.random.normal(loc=0, scale=0.000001, size=1)) #0.00001



    T_values = np.array(T_values, dtype=np.float32)
    real_imag_data = np.array(real_imag_data)
    return T_values, real_imag_data, file_list, time_bins_ref, freq_ref, test_Ts, T_min, T_max

def build_real_imag_model(input_shape, output_shape, l2_reg=0, dropout_rate=0, leaky_relu_alpha=0.1): #0.3
    T_input = layers.Input(shape=input_shape, name='T_input')
    #128
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
    return tf.reduce_mean(tf.abs(y_pred - y_true)) #tf.reduce_mean((tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon))) #tf.reduce_mean(tf.abs(y_pred - y_true)) #tf.reduce_mean((tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon)))#tf.max(y_pred - y_true)#tf.reduce_mean((tf.math.log(1.0 + tf.abs(y_pred - y_true) + epsilon)))

def main():
    # -----------------------
    # 1. Load data
    # -----------------------
    data_directory = '/Volumes/T7 Shield/NJOY/spectrograms/full_range/3x45551'

    T, real_imag, file_list, time_bins_ref, freq_ref, test_Ts, T_min, T_max = load_data_from_h5(data_directory)
    print(f"Files processed: {len(file_list)}")
    print("Real+Imag shape:", real_imag.shape)

    N, h, w, c = real_imag.shape 
    num_pixels = h * w * c 
    real_imag_flat = real_imag.reshape(N, num_pixels)

    scaler_T = StandardScaler()
    #T_norm = scaler_T.fit_transform(np.log(T))

    scaler_T_1 = StandardScaler()
    scaler_T_2 = StandardScaler()
    scaler_T_3 = StandardScaler()
    scaler_T_4 = StandardScaler()
    T_1 = scaler_T_1.fit_transform(T)
    T_2 = scaler_T_2.fit_transform(np.log(T))
    T_3 = scaler_T_3.fit_transform(1 / T)
    T_4 = scaler_T_4.fit_transform(T**3)
    T_norm = np.hstack([
        T_1,
        T_2,      # log(T + 1)
        T_3,     # inverse
        T_4
    ])

    scaler_spec = StandardScaler() 
    real_imag_scaled_flat = (scaler_spec.fit_transform(real_imag_flat))
    real_imag_scaled = real_imag_scaled_flat.reshape((N, h, w, c))

    # -----------------------
    # 4. Train/test split
    # -----------------------
    T_train, T_test, spec_train, spec_test = train_test_split(
        T_norm, real_imag_scaled, test_size=0.000001, random_state=42
    )

    # -----------------------
    # 5. Build model
    # -----------------------
    #input_shape=(1,)
    model = build_real_imag_model(input_shape=(4,), output_shape=(h, w, c))
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
    checkpoint = ModelCheckpoint('best_model_real_imag.keras', monitor='val_loss', save_best_only=True, verbose=1)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # -----------------------
    # 8. Train
    # -----------------------
    history = model.fit(
        T_train, spec_train,
        epochs=150,
        batch_size=16,
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

        #new_T_norm = scaler_T.transform([[np.log(new_T)]])  

        T_1 = scaler_T_1.transform(np.array([[new_T]])) 
        T_2 = scaler_T_2.transform(np.log(np.array([[new_T]]))) 
        T_3 = scaler_T_3.transform(1 / np.array([[new_T]]))
        T_4 = scaler_T_4.transform(np.array([[new_T**3]]))
        new_T_norm = np.hstack([
            T_1,
            T_2,      # log(T + 1)
            T_3,     # inverse
            T_4
        ])




        pred_scaled = model.predict(new_T_norm) 
        pred_scaled = pred_scaled.squeeze()   

        #with h5py.File(f"3x45551_{T_min}_{T_max}_spec_scalers.h5", "w") as hf:
            #hf.create_dataset("spec_scale", data=scaler_spec.scale_, dtype="float32")
            #hf.create_dataset("spec_mean", data=scaler_spec.mean_, dtype="float32")
            #hf.create_dataset("T_scale", data=scaler_T.scale_, dtype="float32")
            #hf.create_dataset("T_mean", data=scaler_T.mean_, dtype="float32")

        # Inverse scale
        pred_scaled_flat = (pred_scaled.reshape(1, -1)) 
        pred_unscaled_flat = scaler_spec.inverse_transform(pred_scaled_flat)
        pred_unscaled = pred_unscaled_flat.reshape(h, w, c) 

        pred_complex = pred_unscaled[...,0] + 1j * pred_unscaled[...,1]

        # -----------------------
        # 11. Save results to HDF5
        # -----------------------
        output_filename = f"../../ML_prediction/predicted_spectrogram_{new_T}_real_imag.h5"
        with h5py.File(output_filename, "w") as h5f:
            h5f.create_dataset("time_bins", data=time_bins_ref)
            h5f.create_dataset("frequencies", data=freq_ref)
            h5f.create_dataset("spectrogram_real", data=pred_unscaled[...,0])
            h5f.create_dataset("spectrogram_imag", data=pred_unscaled[...,1])
        print(f"Saved predicted real+imag spectrogram + axes to {output_filename}")

if __name__ == "__main__":
    main()