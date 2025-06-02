# hyperparam_search.py  ── run with:  python hyperparam_search.py --data_dir path/to/specs
import os, argparse, math, glob, re, h5py, numpy as np, keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1.  Data-loading helper that can inject *hp*-controlled noise
# ---------------------------------------------------------------------
def load_data_with_noise(data_dir, hp):
    σT   = hp.Float("sigma_T",   0.0, 1e-3, sampling="log")
    σS   = hp.Float("sigma_spec",0.0, 1e-2, sampling="log")
    reps = hp.Int( "n_noise_reps", 1, 5)

    T_raw, spec_raw = [], []
    time_ref, freq_ref = None, None

    for fp in glob.glob(os.path.join(data_dir, "spectrogram_T_*.h5")):
        m = re.search(r"_T_([\d\.]+)\.h5$", os.path.basename(fp))
        if not m:          # skip non-matching filenames
            continue
        T_val = float(m.group(1))
        with h5py.File(fp, "r") as h5f:
            r = h5f["spectrogram_real"][:]
            i = h5f["spectrogram_imag"][:]
            if time_ref is None:
                time_ref = h5f["time_bins"][:]
                freq_ref = h5f["frequencies"][:]
        spec = np.stack([r, i], axis=-1)

        for _ in range(reps):
            T_raw.append(T_val + np.random.normal(0, σT))
            noisy_spec = spec + np.abs(spec) * np.random.normal(0, σS, size=spec.shape)
            spec_raw.append(noisy_spec)

    T_arr   = np.array(T_raw, dtype=np.float32).reshape(-1, 1)
    spec_arr= np.array(spec_raw, dtype=np.float32)
    return T_arr, spec_arr, time_ref, freq_ref


# ---------------------------------------------------------------------
# 2.  Model-builder for Keras-Tuner
# ---------------------------------------------------------------------
def model_builder(hp, output_shape):
    l2 = hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")
    α  = hp.Float("alpha",  0.01, 0.4, sampling="linear")

    # choose up to 3 hidden layers
    n_hidden = hp.Int("n_hidden", 1, 4)
    units = [hp.Int(f"units_{k}", min_value=16, max_value=512, step=32)
             for k in range(n_hidden)]

    latent_units = hp.Choice("latent_units", [4, 8, 16])

    inp = layers.Input(shape=(1,), name="T_input")
    x   = inp
    for u in units:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2))(x)
        x = LeakyReLU(alpha=α)(x)

    x = layers.Dense(latent_units,
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = LeakyReLU(alpha=α)(x)

    flat = np.prod(output_shape)
    out  = layers.Dense(flat, activation="linear")(x)
    out  = layers.Reshape(output_shape)(out)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=lambda y,t: tf.reduce_mean(tf.abs(t-y)),
                  metrics=["mae"])
    return model


# ---------------------------------------------------------------------
# 3.  Tuner
# ---------------------------------------------------------------------
def run_search(data_dir, max_trials=40, executions=1):
    # dummy read to get H×W×C
    any_h5 = glob.glob(os.path.join(data_dir, "spectrogram_T_*.h5"))[0]
    with h5py.File(any_h5, "r") as h5f:
        H,W   = h5f["spectrogram_real"].shape
    C = 2
    output_shape = (H, W, C)

    def _build(hp):
        return model_builder(hp, output_shape)

    tuner = kt.BayesianOptimization(
        hypermodel=_build,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        executions_per_trial=executions,
        directory="tuner_runs",
        project_name="spec_MLP"
    )

    # -----------------------------------------------------------------
    # Data once, inside the tuner’s search-space loop
    # -----------------------------------------------------------------
    for trial_hp in tuner._split_trials():
        T, spec, *_ = load_data_with_noise(data_dir, trial_hp)
        # scale inside each trial so σ ranges are comparable
        T_sc   = StandardScaler().fit_transform(T)
        flat   = spec.reshape(len(spec), -1)
        spec_sc= StandardScaler().fit_transform(flat).reshape(spec.shape)

        T_tr, T_val, S_tr, S_val = train_test_split(
            T_sc, spec_sc, test_size=0.15, random_state=42
        )

        tuner._queue.put((trial_hp, (T_tr, S_tr), (T_val, S_val)))

    tuner.search()
    return tuner


# -------------  CLI  -------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with spectrogram_T_*.h5")
    p.add_argument("--trials", type=int, default=80)
    args = p.parse_args()

    best_tuner = run_search(args.data_dir, max_trials=args.trials)
    print("Best val_loss:", best_tuner.oracle.get_best_trials(1)[0].score)
    best_hp = best_tuner.get_best_hyperparameters()[0]
    print("Best hyper-parameters:")
    for k,v in best_hp.values.items():
        print(f"  {k}: {v}")
