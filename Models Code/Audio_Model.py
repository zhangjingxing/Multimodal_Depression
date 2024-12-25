import librosa
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed, LSTM
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import random

# Path to audio files
audio_directory = 'drive/MyDrive/ML data/clean audio/all data' #################################################################

# Load Excel file
file_path = 'drive/MyDrive/ML data/Phq8_data.xlsx'
df = pd.read_excel(file_path)

# Extract participant IDs and their labels
participant_ids = df['Participant_ID'].astype(str).values  # Convert to string
labels = df['PHQ8_Binary'].values  # Target labels (1: Depression, 0: No depression)

# ==============================
# 2. Match Audio Files with Labels
# ==============================
audio_files = sorted([file for file in os.listdir(audio_directory) if file.endswith('.wav')])

matched_files = []
matched_labels = []

for participant_id, label in zip(participant_ids, labels):
    file_name = f"{participant_id}.wav"
    matched_file = next((f for f in audio_files if participant_id in f), None)
    if matched_file:
        matched_files.append(os.path.join(audio_directory, matched_file))
        matched_labels.append(label)
    else:
        print(f"File not found: {file_name}")

# ==============================
# 3. Define MFCC Extraction and Framing Function
# ==============================
"""
    Load an audio file, compute its normalized MFCC, add noise (augmentation), pad it to ensure
    it can be evenly divided into frames of size `frame_size`, split it into frames,
    and organize frames into sequences of `sequence_length`.

    Parameters:
    - file_path (str): Path to the audio file.
    - sr (int): Sampling rate for loading the audio.
    - n_mfcc (int): Number of MFCC coefficients to extract.
    - hop_length (int): Number of samples between successive frames.
    - frame_size (int): Number of time steps per frame.
    - sequence_length (int): Number of frames per sequence.
    - noise_factor (float): Noise factor for augmentation.

    Returns:
    - sequences (np.ndarray): Array of MFCC sequences with shape
                              (num_sequences, sequence_length, n_mfcc, frame_size, 1).
    """
def compute_mfcc_sequences(file_path, sr=16000, n_mfcc=13, hop_length=512, frame_size=100, sequence_length=10, noise_factor=0.005):
    try:

        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr)

        # 2. Apply data augmentation techniques
        # a. Add random noise
        audio = audio + noise_factor * np.random.randn(len(audio))

        # b. Time-stretching with a random factor
        stretch_factor = np.random.uniform(0.8, 1.2)  # Stretching factor
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)

        # c. Pitch-shifting within a small range
        pitch_steps = np.random.randint(-2, 2)  # Steps to shift
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)

        # d. Random trimming to vary start and end points
        max_offset = len(audio) // 10
        start_offset = np.random.randint(0, max_offset)
        end_offset = np.random.randint(0, max_offset)
        audio = audio[start_offset:len(audio) - end_offset]

        # e. Dynamic range compression (simulate loudness variations)
        audio = librosa.effects.percussive(audio, margin=3)

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

# ==============================
# 4. Extract Features from Audio Files
# ==============================
n_mfcc, frame_size, sequence_length, hop_length = 13, 100, 10, 512

all_sequences, all_labels_seq = [], []

for i, file_path in enumerate(matched_files):
    sequences = compute_mfcc_sequences(file_path, sr=16000, n_mfcc=n_mfcc, hop_length=hop_length, frame_size=frame_size, sequence_length=sequence_length)
    if sequences is not None:
        all_sequences.append(sequences)
        all_labels_seq.extend([matched_labels[i]] * sequences.shape[0])

# Convert to NumPy arrays
all_sequences = np.vstack(all_sequences)
all_labels_seq = np.array(all_labels_seq)

print(f"Total sequences: {all_sequences.shape[0]}")
print(f"Labels shape: {all_labels_seq.shape}")

# ==============================
# 5. Train-Test Split
# ==============================
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    all_sequences, all_labels_seq, test_size=0.2, random_state=42, stratify=all_labels_seq
)

print(f"Training samples: {X_train_seq.shape[0]}")
print(f"Testing samples: {X_test_seq.shape[0]}")

# ==============================
# 6. Define CNN-RNN Model
# ==============================
input_shape_seq = X_train_seq.shape[1:]
cnn_input_seq = Input(shape=input_shape_seq)

# TimeDistributed CNN layers
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))(cnn_input_seq)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))(x)
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

x = TimeDistributed(Flatten())(x)

# RNN layers
x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
x = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)
x = Dropout(0.5)(x)

# Output layer
output_seq = Dense(1, activation='sigmoid')(x)

# Compile the model
model_seq = Model(inputs=cnn_input_seq, outputs=output_seq)
model_seq.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
model_seq.summary()

# ==============================
# 7. Train the Model without EarlyStopping
# ==============================
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

history_seq = model_seq.fit(
    X_train_seq, y_train_seq, epochs=30, batch_size=16, validation_data=(X_test_seq, y_test_seq), verbose=1,
    callbacks=[lr_reduction]
)

# ==============================
# 8. Evaluate the Model
# ==============================
loss_seq, accuracy_seq = model_seq.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"Test Loss: {loss_seq:.4f}, Test Accuracy: {accuracy_seq:.4f}")

# ==============================
# 12. Plot Training History
# ==============================

# Plot training & validation accuracy and loss for sequential model
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_seq.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history_seq.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Sequential Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history_seq.history['loss'], label='Train Loss', color='blue')
plt.plot(history_seq.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Sequential Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
