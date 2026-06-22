import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pathlib

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Load your Excel Sheets ---
try:
    train_df = pd.read_excel('Training_data.xlsx')
    test_df = pd.read_excel('Testing_data.xlsx')
    print("Successfully loaded training_data.xlsx and testing_data.xlsx")
except FileNotFoundError:
    print("Error: Please make sure your Excel files are in this folder.")
    exit()

# --- 2. Convert Excel Columns Directly to NumPy Arrays (No Scaling!) ---
# Inputs: Voltage, Current, Temperature
x_train = train_df[['Voltage_V', 'Current_A', 'Temperature_C']].values
x_test = test_df[['Voltage_V', 'Current_A', 'Temperature_C']].values

# Outputs: State of Charge (SoC), State of Health (SoH)
y_train = train_df[['SoC_percent', 'SoH_percent']].values
y_test = test_df[['SoC_percent', 'SoH_percent']].values

layer_1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(3,))
layer_2 = tf.keras.layers.Dense(32, activation='relu')
layer_3 = tf.keras.layers.Dense(16, activation='relu')
layer_4 = tf.keras.layers.Dense(8, activation='relu')
layer_5 = tf.keras.layers.Dense(2)

model = tf.keras.Sequential([layer_1, layer_2, layer_3, layer_4, layer_5])

quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(model)
q_aware_model.compile(optimizer='adam', loss='mean_squared_error')

q_aware_model.fit(x_train, y_train, epochs=1000, batch_size=16, verbose='auto')

predicted_values = q_aware_model.predict(x_test)
actual_values = y_test
print(f"Predicted values are = {predicted_values}")
print(f"Actual values are = {actual_values}")

model_path = "Final_model"
tf.saved_model.save(q_aware_model, model_path)
size_bytes = os.path.getsize(model_path)
size_kb = size_bytes / 1024
size_mb = size_bytes / (1024 * 1024)
print(f"Size: {size_bytes:.2f} B")
print(f"Size: {size_kb:.2f} KB")
print(f"Size: {size_mb:.2f} MB")
print(q_aware_model.summary())

representative_data = x_train[::100].astype(np.float32)


def representative_dataset_gen():
    for input_value in representative_data:
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)  # Changed from from_saved_model
# converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
tflite_model_file = pathlib.Path('lite/model.tflite')
tflite_model_file.parent.mkdir(parents=True, exist_ok=True)
size_opti_model = tflite_model_file.write_bytes(tflite_model)
print(f"Size of optimized model = {size_opti_model}")
print(f"Size of optimized model in KB= {size_opti_model / 1024.0}")
print(f"Size of optimized model in MB= {size_opti_model / (1024.0 * 1024.0)}")

y_pred = q_aware_model.predict(x_test, verbose=0)

error = np.abs(y_test - y_pred)

tolerance = 4.5

accuracy = np.mean(error <= tolerance) * 100

print(f"TensorFlow Accuracy: {accuracy:.2f}%")
print(f"Max Error: {np.max(error):.6f}")
print(f"Mean Error: {np.mean(error):.6f}")


def tflite_predict_simple(interpreter, x_test):
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    predictions = []
    # Use a standard for-loop for clarity and reliability
    for x in x_test:
        # Set the input tensor
        interpreter.set_tensor(input_idx, np.array([x], dtype=np.float32))
        # Run the model
        interpreter.invoke()
        # Get the output and add it to the list
        prediction = interpreter.get_tensor(output_idx).squeeze()
        predictions.append(prediction)

    return np.array(predictions)


interpreter = tf.lite.Interpreter(model_path=str(pathlib.Path(tflite_model_file)))
interpreter.allocate_tensors()

print('Input details:')
print(interpreter.get_input_details())
print('\nOutput details:')
print(interpreter.get_output_details())
y_pred = tflite_predict_simple(interpreter, x_test)
print("First 10 expected:")
print(y_test[:10])

print("First 10 TFLite predictions:")
print(y_pred[:10])
error = np.abs(y_test - y_pred)

tolerance = 4.5

accuracy = np.mean(error <= tolerance) * 100

print(f"TFLite Accuracy: {accuracy:.2f}%")
print(f"Max Error: {np.max(error):.6f}")
print(f"Mean Error: {np.mean(error):.6f}")

import os

tflite_path = "lite/model.tflite"

# Read the binary bytes of your optimized model
with open(tflite_path, "rb") as f:
    tflite_model = f.read()

# Convert the raw bytes into a C++ hexadecimal format
hex_lines = [", ".join([f"0x{b:02x}" for b in tflite_model[i:i + 12]]) for i in range(0, len(tflite_model), 12)]
c_array = ",\n  ".join(hex_lines)

# Write out the C++ header file
with open("model.h", "w") as f:
    f.write('// Auto-generated QAT TensorFlow Lite Model Array\n\n')
    f.write('#ifndef MODEL_H\n#define MODEL_H\n\n')
    f.write('const unsigned char sine_model[] = {\n  ' + c_array + '\n};\n\n')
    f.write(f'const unsigned int sine_model_len = {len(tflite_model)};\n\n')
    f.write('#endif // MODEL_H\n')

print("C++ model.h file successfully generated!")