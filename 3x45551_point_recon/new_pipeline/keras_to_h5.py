from tensorflow import keras
import keras

# 1. Load your .keras model
model = keras.models.load_model('3x45551_950_1050.keras')

# 2. Re-save it in HDF5 format
model.save('3x45551_950_1050.h5', save_format='h5')