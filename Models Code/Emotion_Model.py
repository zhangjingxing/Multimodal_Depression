

# Import libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import kagglehub
#save them
SAVEE = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")
RAV = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
TESS =  kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
CREMA = kagglehub.dataset_download("ejlok1/cremad")


SAVEE = os.path.join(SAVEE, "ALL")
RAV = os.path.join(RAV, "audio_speech_actors_01-24")
TESS = os.path.join(TESS, "tess toronto emotional speech set data","TESS Toronto emotional speech set data")
CREMA = os.path.join(CREMA, "AudioWAV")

# Get the data location for SAVEE
dir_list = os.listdir(SAVEE)

# parse the filename to get the emotions
emotion=[]
path = []
for i in dir_list:
    if i[-8:-6]=='_a':
        emotion.append('male_angry')
    elif i[-8:-6]=='_d':
        emotion.append('male_disgust')
    elif i[-8:-6]=='_f':
        emotion.append('male_fear')
    elif i[-8:-6]=='_h':
        emotion.append('male_happy')
    elif i[-8:-6]=='_n':
        emotion.append('male_neutral')
    elif i[-8:-6]=='sa':
        emotion.append('male_sad')
    elif i[-8:-6]=='su':
        emotion.append('male_surprise')
    else:
        emotion.append('male_error')
    #path.append(SAVEE + i)
    file_path = os.path.join(SAVEE, i)
    path.append(file_path)

# Now check out the label count distribution
SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
SAVEE_df.labels.value_counts()

dir_list = os.listdir(RAV)
dir_list.sort()

emotion = []
gender = []
path = []
for i in dir_list:
    actor_folder = os.path.join(RAV, i)
    fname = os.listdir(actor_folder)
    #fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        #path.append(RAV + i + '/' + f)
        path.append(os.path.join(actor_folder, f))


RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'
RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
RAV_df.labels.value_counts()

dir_list = os.listdir(TESS)
dir_list.sort()
dir_list

path = []
emotion = []

for i in dir_list:
    folder_path = os.path.join(TESS, i)  # Correctly join the folder path
    fname = os.listdir(folder_path)  # List files in the current folder
    #fname = os.listdir(TESS + i)
    for f in fname:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotion.append('female_angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion.append('female_disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion.append('female_fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion.append('female_happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion.append('female_neutral')
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion.append('female_surprise')
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion.append('female_sad')
        else:
            emotion.append('Unknown')
        file_path = os.path.join(folder_path, f)  # Correctly join file path
        path.append(file_path)
        #path.append(TESS + i + "/" + f)

TESS_df = pd.DataFrame(emotion, columns = ['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
TESS_df.labels.value_counts()

dir_list = os.listdir(CREMA)
dir_list.sort()
print(dir_list[0:10])

gender = []
emotion = []
path = []
female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

for i in dir_list:
    part = i.split('_')
    # Check if the first part is a valid speaker ID (integer)
    try:
        speaker_id = int(part[0])
    except ValueError:
        # Skip files with unexpected naming format
        print(f"Skipping file with unexpected format: {i}")
        continue
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion.append('male_sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotion.append('male_angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotion.append('male_disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotion.append('male_fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotion.append('male_happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotion.append('male_neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotion.append('female_sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotion.append('female_angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotion.append('female_disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotion.append('female_fear')
    elif part[2] == 'HAP' and temp == 'female':
        emotion.append('female_happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotion.append('female_neutral')
    else:
        emotion.append('Unknown')
    path.append(file_path)

CREMA_df = pd.DataFrame(emotion, columns = ['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
CREMA_df.labels.value_counts()

df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)
print(df.labels.value_counts())
df.head()
df.to_csv("Data_path.csv",index=False)

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Flatten, TimeDistributed, LSTM, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the consolidated dataset
df = pd.read_csv("Data_path.csv")

# Display label distribution
print(df['labels'].value_counts())

# Preview the first few entries
print(df.head())


# Function to extract emotion from the label
def extract_emotion(label):
    # Split the label by '_' and take the second part (emotion)
    return label.split('_')[1] if '_' in label else label

# Apply the function to create a new 'emotion' column
df['emotion'] = df['labels'].apply(extract_emotion)

# Display the new label distribution
print("\nLabel Distribution After Combining:")
print(df['emotion'].value_counts())

# Preview the updated DataFrame
print("\nUpdated DataFrame:")
print(df.head())

# Save all paths to a CSV file
df[['path']].to_csv("all_file_paths.csv", index=False)
print("\nAll file paths have been saved to 'all_file_paths.csv'.")

import os

# Function to check if a file exists
def check_file_exists(file_path):
    return os.path.exists(file_path)

# Apply the function to create a new 'file_exists' column
df['file_exists'] = df['path'].apply(check_file_exists)

# Display the count of existing and missing files
print("\nFile Existence Check:")
print(df['file_exists'].value_counts())

import os

# Function to check if a file exists
def check_file_exists(file_path):
    return os.path.exists(file_path)

# Apply the function to create a new 'file_exists' column
df['file_exists'] = df['path'].apply(check_file_exists)

# Display the count of existing and missing files
print("\nFile Existence Check:")
print(df['file_exists'].value_counts())

!pip install keras
!pip install tensorflow

# ==============================
# Importing Required Libraries
# ==============================

# TensorFlow and Keras imports using TensorFlow's Keras
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D  # Added Conv2D layers
from tensorflow.keras.utils import to_categorical  # Removed np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop  # Imported RMSprop directly

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other Libraries
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os
import pickle
import IPython.display as ipd  # To play sound in the notebook

# ============================
# Label Encoding and Setup
# ============================

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Assuming you have a DataFrame `df` with columns 'path' and 'emotion'
# If not, you need to create/load it accordingly
# Example:
# df = pd.read_csv('your_dataset.csv')  # Ensure it has 'path' and 'emotion' columns

# Fit and transform the 'emotion' labels
df['label_encoded'] = label_encoder.fit_transform(df['emotion'])

# Number of classes
num_classes = df['label_encoded'].nunique()
print(f"\nNumber of Classes: {num_classes}")

# Display the mapping from labels to encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# ================================
# Data Validation and Filtering
# ================================

# Define parameters for validation
frame_size = 100  # Adjust based on your requirements
sequence_length = 10  # Adjust based on your requirements

# Debugging: Filter invalid or too-short audio files
valid_files = []
for path in df['path']:
    try:
        y, sr = librosa.load(path, sr=16000)  # Ensure the sample rate is consistent
        if len(y) >= frame_size * sequence_length:
            valid_files.append(path)
        else:
            print(f"File {path} is too short for processing.")
    except Exception as e:
        print(f"Error reading file {path}: {e}")

# Filter the DataFrame to include only valid files
df = df[df['path'].isin(valid_files)].reset_index(drop=True)
print(f"Filtered dataset size: {df.shape}")

# ==================================
# MFCC Feature Extraction Function
# ==================================

def compute_mfcc_sequences(file_path, sr=16000, n_mfcc=13, hop_length=512, frame_size=100, sequence_length=10):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        normalized_mfcc = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))

        # Pad MFCC to fit sequence and frame requirements
        num_time_frames = normalized_mfcc.shape[1]
        total_required_frames = frame_size * sequence_length
        remainder = num_time_frames % total_required_frames
        if remainder != 0:
            padding_width = total_required_frames - remainder
            normalized_mfcc = np.pad(normalized_mfcc, ((0, 0), (0, padding_width)), mode='constant')
            print(f"Padded MFCC with {padding_width} zeros for {file_path}.")

        # Split into frames and sequences
        num_frames = normalized_mfcc.shape[1] // frame_size
        frames = [normalized_mfcc[:, i * frame_size:(i + 1) * frame_size] for i in range(num_frames)]
        frames = np.array(frames)[..., np.newaxis]  # Add channel dimension
        num_sequences = frames.shape[0] // sequence_length
        if num_sequences == 0:
            print(f"File {file_path} too short for the desired sequence length.")
            return None

        frames = frames[:num_sequences * sequence_length]
        sequences = frames.reshape(num_sequences, sequence_length, n_mfcc, frame_size, 1)
        return sequences

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# ============================
# Feature Extraction Pipeline
# ============================

# Parameters
n_mfcc = 13
hop_length = 512
frame_size = 100
sequence_length = 10

all_sequences, all_labels_seq = [], []

# Iterate through each file in the dataset
for idx, row in df.iterrows():
    file_path = row['path']
    label = row['label_encoded']

    sequences = compute_mfcc_sequences(
        file_path,
        sr=16000,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        frame_size=frame_size,
        sequence_length=sequence_length
    )

    if sequences is not None:
        all_sequences.append(sequences)
        all_labels_seq.extend([label] * sequences.shape[0])

    # Optional: Print progress every 500 files
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(df)} files.")

# Convert to NumPy arrays
if all_sequences:
    all_sequences = np.vstack(all_sequences)
    all_labels_seq = np.array(all_labels_seq)

    print(f"\nTotal sequences: {all_sequences.shape[0]}")
    print(f"Sequences shape: {all_sequences.shape}")
    print(f"Labels shape: {all_labels_seq.shape}")
else:
    print("No sequences were extracted. Please check your data and MFCC extraction parameters.")

# ============================
# Preparing Data for Modeling
# ============================

# Check if sequences were extracted
if all_sequences.size == 0:
    raise ValueError("No data available for training. Exiting the script.")

#Convert labels to categorical (one-hot encoding)
y = to_categorical(all_labels_seq, num_classes=num_classes)
print(f"\nOne-hot Encoded Labels Shape: {y.shape}")

# Split between train and test
# Here, each sequence is treated as an independent sample
X_train, X_test, y_train, y_test = train_test_split(
    all_sequences,
    y,
    test_size=0.25,
    shuffle=True,
    random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ============================
# Data Normalization
# ============================

# Normalize data: Standardization (mean=0, std=1)
mean = np.mean(X_train, axis=(0, 1, 2, 3, 4))
std = np.std(X_train, axis=(0, 1, 2, 3, 4))

X_train = (X_train - mean) / (std + 1e-6)
X_test = (X_test - mean) / (std + 1e-6)

# Further normalization if needed
X_train = X_train - np.mean(X_train)
X_test = X_test - np.mean(X_test)

print(f"Normalized Training data range: {X_train.min()} to {X_train.max()}")
print(f"Normalized Testing data range: {X_test.min()} to {X_test.max()}")

# ============================
# Model Definition
# ============================

# Assuming the input shape is (sequence_length, n_mfcc, frame_size, 1)
# We need to adjust the model to handle this 5D input, possibly using TimeDistributed layers
# Alternatively, reshape the data to 3D by treating the sequence as part of the feature dimension

# For simplicity, let's reshape the data to (samples, sequence_length * n_mfcc, frame_size, 1)
# and use Conv2D instead of Conv1D

# Reshape data
X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length * n_mfcc, frame_size, 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length * n_mfcc, frame_size, 1)

print(f"Reshaped Training data shape: {X_train_reshaped.shape}")
print(f"Reshaped Testing data shape: {X_test_reshaped.shape}")

# Define the model
model = Sequential()

# First Conv2D layer
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(sequence_length * n_mfcc, frame_size, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Second Conv2D layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Third Conv2D layer
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# 4hh Conv2D layer
#model.add(Conv2D(512, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
# Updated optimizer with correct parameter names
opt = RMSprop(learning_rate=0.003)  # Removed 'decay' and updated 'lr' to 'learning_rate'

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# ============================
# Model Training
# ============================

# Define callbacks
checkpoint = ModelCheckpoint(
    filepath='saved_models/Emotion_Model.keras',  # Changed extension to .keras
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Optional: Learning Rate Scheduler Callback (e.g., ReduceLROnPlateau)
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    factor=0.5,
    min_lr=1e-7
)

callbacks_list = [checkpoint, lr_reduction]

# Train the model
model_history = model.fit(
    X_train_reshaped,
    y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test_reshaped, y_test),
    callbacks=callbacks_list
)

# ============================
# Saving the Model
# ============================

# Save the entire model to a .keras file
model_name = 'Emotion_Model.keras'  # Changed to .keras
save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'Saved model and weights at {model_path}')

# Save the model architecture to JSON
model_json = model.to_json()
with open(os.path.join(save_dir, "model_json.json"), "w") as json_file:
    json_file.write(model_json)

print("Saved model architecture to JSON.")

# ============================
# Loading and Evaluating Model
# ============================

# Load JSON and create model
with open(os.path.join(save_dir, 'model_json.json'), 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load weights into the new model
loaded_model.load_weights(model_path)
print("Loaded model from disk")

# Compile the loaded model
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Evaluate the loaded model
score = loaded_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"{loaded_model.metrics_names[1]}: {score[1]*100:.2f}%")

# ============================
# Optional: Plot Training History
# ============================

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'], label='Train Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ============================
# Optional: Confusion Matrix
# ============================

# Predict classes
y_pred = loaded_model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))


# Save mean and std
np.save('saved_models/mean.npy', mean)
np.save('saved_models/std.npy', std)

# Save the label encoder
import pickle
with open('saved_models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)