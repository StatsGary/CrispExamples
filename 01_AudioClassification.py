# =============================================================================
# Audio Classification with Tensorflow 
# =============================================================================
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed for reproducibility of results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# =============================================================================
# # Import the speech command dataset
# =============================================================================
data_dir = pathlib.Path('data/mini_speech_commands')

# Create the data directory if it does not exist
if not data_dir.exists():
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')
    
    
# =============================================================================
# # Shuffle the dataset
# =============================================================================

def shuffle_dataset(data_dir):
    # Check basic statistics about the dataset
    global commands
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    #Exclude the README file from the directory structure
    commands = commands[commands!="README.md"]
    print("Commands:", commands)
    global filenames #Make function global as it needs to be used later
    filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print("Total number of examples is: ", num_samples)
    print("Number of examples per label:",
          len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print("Example file tensor:", filenames[0])
    
    
shuffle_dataset(data_dir)
    
# =============================================================================
# Split the dataset
# =============================================================================

def split_train_test_val(data, split_prop):
    file_len = len(data)
    train_samp = int(file_len * split_prop)
    remain_samp = file_len - train_samp
    test_size = int(remain_samp / 2)
    val_size = int(remain_samp / 2)
    check = file_len - train_samp - test_size - val_size
    if check > 0:
        raise Exception("The proportion calculations are wrong, please enter a whole number")
    # Create splits
    train_files = data[:train_samp]
    val_files = data[train_samp: train_samp + val_size]
    test_files = data[-test_size:]
    #Return a tuple of files
    print("Training set size is :", train_samp)
    print("Testing set size is: ", test_size)
    return (train_files, val_files, test_files)

# Use function to create training, validation and testing sets
train_files, val_files, test_files = split_train_test_val(filenames, 0.8)

# =============================================================================
# Reading audio files and their labels
# =============================================================================
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)
#The sample rate for this dataset is 16kHz. Note that tf.audio.decode_wav will normalize the values to the range [-1.0, 1.0].

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# =============================================================================
# EDA
# =============================================================================
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()

# =============================================================================
# Spectogram
# =============================================================================

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  return spectrogram

# Comparing waveform and the actual audio
for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print ('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

# Now transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs.

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id


spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# Examine the spectrogram "images" for different samples of the dataset.

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
  ax.set_title(commands[label_id.numpy()])
  ax.axis('off')

plt.show()

# =============================================================================
# Build and Train the model
# =============================================================================
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

# Batch the training and validation sets for model training
batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

#Add dataset cache() and prefetch() operations to reduce read latency while training the model.
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# =============================================================================
# # Train a CNN model as the spectograms have been converted as images
# =============================================================================

for spectogram, _ in spectrogram_ds.take(1):
    input_shape = spectogram.shape
print("Input shape:", input_shape)
number_of_labels = len(commands)

# =============================================================================
# Create Model
# =============================================================================
# Add a normalisation layer
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

# Create modelling layer
model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer, 
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64,3, activation='relu'),
    layers.MaxPooling2D(), 
    layers.Dropout(0.25), 
    layers.Flatten(), 
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5), 
    layers.Dense(number_of_labels)
    ])

# Grab a summary of the model
model.summary()

# =============================================================================
# Compile Model
# =============================================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# =============================================================================
# Fit the model
# =============================================================================
EPOCHS = 50
history = model.fit(
    train_ds, 
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(patience=3))

# =============================================================================
# Monitor the model
# =============================================================================
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

# =============================================================================
# Evaluate the test set performance
# =============================================================================
#Runt the model on our test set to check perform
test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# Make a prediction and sum up the accuracy and take argmax
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

# =============================================================================
# Make a confusion matrix
# =============================================================================
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# =============================================================================
# Make an inference / prediction on the audio files
# =============================================================================

sample_pred = data_dir/'no/01bb6a2a_nohash_0.wav'
sample_ds = preprocess_dataset([str(sample_pred)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{commands[label[0]]}"')
  plt.show()
