# Import necessary libraries
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, TimeDistributed, LSTM
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ======================================
# Step 1: Load and Preprocess Audio Data
# ======================================
# Paths to audio files and PHQ8 data
audio_directory = 'drive/MyDrive/ML data/clean audio/all data'
phq8_file_path = 'drive/MyDrive/ML data/Phq8_data.xlsx'

# Load PHQ8 data
df_phq8 = pd.read_excel(phq8_file_path)

# Filter participant IDs between 300 and 492
df_phq8 = df_phq8[(df_phq8['Participant_ID'] >= 300) & (df_phq8['Participant_ID'] <= 492)]

# Extract participant IDs and their labels
participant_ids = df_phq8['Participant_ID'].astype(str).values  # Convert to string
labels = df_phq8['PHQ8_Binary'].values  # Target labels (1: Depression, 0: No depression)

# Create a participant label dictionary
participant_label_dict = dict(zip(participant_ids, labels))

# Match audio files with labels
audio_files = sorted([file for file in os.listdir(audio_directory) if file.endswith('.wav')])

matched_files = []
matched_labels = []
matched_participant_ids = []

for participant_id, label in zip(participant_ids, labels):
    file_name = f"{participant_id}.wav"
    matched_file = next((f for f in audio_files if participant_id in f), None)
    if matched_file:
        matched_files.append(os.path.join(audio_directory, matched_file))
        matched_labels.append(label)
        matched_participant_ids.append(participant_id)
    else:
        print(f"Audio file not found: {file_name}")

# Ensure that we have data for audio modality
if not matched_files:
    raise ValueError("No audio files were matched. Please check your audio directory and participant IDs.")

# =========================================
# Step 2: Define MFCC Extraction Function
# =========================================
def compute_mfcc_sequences(file_path, sr=16000, n_mfcc=13, hop_length=512,
                           frame_size=100, sequence_length=10, noise_factor=0.005):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr)

        # Data augmentation techniques
        # a. Add random noise
        audio = audio + noise_factor * np.random.randn(len(audio))

        # b. Time-stretching with a random factor
        stretch_factor = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(audio, rate=stretch_factor)

        # c. Pitch-shifting within a small range
        pitch_steps = np.random.randint(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)

        # d. Random trimming
        max_offset = len(audio) // 10
        start_offset = np.random.randint(0, max_offset)
        end_offset = np.random.randint(0, max_offset)
        audio = audio[start_offset:len(audio) - end_offset]

        # e. Dynamic range compression
        audio = librosa.effects.percussive(audio, margin=3)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        normalized_mfcc = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs) + 1e-6)

        # Pad MFCC to fit sequence and frame requirements
        num_time_frames = normalized_mfcc.shape[1]
        total_required_frames = frame_size * sequence_length
        remainder = num_time_frames % total_required_frames
        if remainder != 0:
            padding_width = total_required_frames - remainder
            normalized_mfcc = np.pad(normalized_mfcc, ((0, 0), (0, padding_width)), mode='constant')

        # Split into frames and sequences
        num_frames = normalized_mfcc.shape[1] // frame_size
        frames = [normalized_mfcc[:, i * frame_size:(i + 1) * frame_size] for i in range(num_frames)]
        frames = np.array(frames)[..., np.newaxis]  # Add channel dimension
        num_sequences = frames.shape[0] // sequence_length
        if num_sequences == 0:
            return None

        frames = frames[:num_sequences * sequence_length]
        sequences = frames.reshape(num_sequences, sequence_length, n_mfcc, frame_size, 1)
        return sequences

    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None

# =========================================
# Step 3: Extract Features from Audio Files
# =========================================
n_mfcc, frame_size, sequence_length, hop_length = 13, 100, 10, 512

all_sequences = []
all_labels_seq = []
all_participant_ids_seq = []

for i, file_path in enumerate(matched_files):
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
        all_labels_seq.extend([matched_labels[i]] * sequences.shape[0])
        all_participant_ids_seq.extend([matched_participant_ids[i]] * sequences.shape[0])

# Ensure that we have sequences
if not all_sequences:
    raise ValueError("No sequences were extracted from audio files.")

# Convert to NumPy arrays
all_sequences = np.vstack(all_sequences)
all_labels_seq = np.array(all_labels_seq)
all_participant_ids_seq = np.array(all_participant_ids_seq)

print(f"Total sequences: {all_sequences.shape[0]}")
print(f"Labels shape: {all_labels_seq.shape}")
print(f"Participant IDs shape: {all_participant_ids_seq.shape}")

# =========================================
# Step 4: Unified Train-Test Split Based on Participant IDs
# =========================================
# Get the list of participants present in all modalities

# Load the transcript data
transcript_file_path = 'drive/MyDrive/ML data/Transcript_Feature.csv'  # Update with your file path
data_transcript = pd.read_csv(transcript_file_path)

# Filter participant IDs between 300 and 492
data_transcript = data_transcript[(data_transcript['Case'] >= 300) & (data_transcript['Case'] <= 492)]

# Convert 'Case' to string
data_transcript['Case'] = data_transcript['Case'].astype(str)
participant_ids_transcript = data_transcript['Case'].values

# Load facial features participant IDs
transcript_features_file = 'drive/MyDrive/ML data/Transcript_Feature.csv'  # Path to your transcript features file
transcript_features = pd.read_csv(transcript_features_file)

# Extract the PHQ8 label and 'Case' (patient ID)
labels_df = transcript_features[['Case', 'PHQ8']].copy()
labels_df['Case'] = labels_df['Case'].astype(int)
labels_df['Case'] = labels_df['Case'].astype(str)

# Load facial features
au_files_path = 'drive/MyDrive/ML data/AUs'  # Replace with your actual AU files folder
participant_ids_facial = []

for file in os.listdir(au_files_path):
    if file.endswith('.txt'):  # Ensure it's an AU feature file
        patient_id = file.split('_')[0]  # Extract patient ID from filename
        participant_ids_facial.append(patient_id)

participant_ids_facial = np.unique(participant_ids_facial)

# Find common participant IDs across all modalities
participant_ids_audio = np.unique(matched_participant_ids)
participant_ids_transcript = np.unique(participant_ids_transcript)
participant_ids_facial = np.unique(participant_ids_facial)

common_participant_ids = np.intersect1d(
    np.intersect1d(participant_ids_audio, participant_ids_transcript),
    participant_ids_facial
)

print(f"Number of common participants across modalities: {len(common_participant_ids)}")

# Create a participant label dictionary for common participants
labels_for_common_ids = [participant_label_dict[pid] for pid in common_participant_ids]

# Perform unified train-test split
ids_train, ids_test = train_test_split(
    common_participant_ids,
    test_size=0.2,
    random_state=42,
    stratify=labels_for_common_ids
)

# =========================================
# Step 5: Split Data for Each Modality Using Unified IDs
# =========================================

# For Audio
train_mask_audio = np.isin(all_participant_ids_seq, ids_train)
test_mask_audio = np.isin(all_participant_ids_seq, ids_test)

X_train_seq = all_sequences[train_mask_audio]
X_test_seq = all_sequences[test_mask_audio]
y_train_seq = all_labels_seq[train_mask_audio]
y_test_seq = all_labels_seq[test_mask_audio]
participant_ids_train_seq = all_participant_ids_seq[train_mask_audio]
participant_ids_test_seq = all_participant_ids_seq[test_mask_audio]

# For Transcript
data_transcript = data_transcript[data_transcript['Case'].isin(common_participant_ids)]
feature_columns = data_transcript.columns.drop(['PHQ8', 'Case'])
X_transcript = data_transcript[feature_columns].values
y_transcript = data_transcript['PHQ8'].values
participant_ids_transcript = data_transcript['Case'].values

train_mask_transcript = np.isin(participant_ids_transcript, ids_train)
test_mask_transcript = np.isin(participant_ids_transcript, ids_test)

X_train_transcript = X_transcript[train_mask_transcript]
X_test_transcript = X_transcript[test_mask_transcript]
y_train_transcript = y_transcript[train_mask_transcript]
y_test_transcript = y_transcript[test_mask_transcript]
participant_ids_train_transcript = participant_ids_transcript[train_mask_transcript]
participant_ids_test_transcript = participant_ids_transcript[test_mask_transcript]

# For Facial Features
# Load and preprocess facial features
all_au_features = []

for file in os.listdir(au_files_path):
    if file.endswith('.txt'):
        file_path = os.path.join(au_files_path, file)
        if os.path.getsize(file_path) > 0:
            patient_id = file.split('_')[0]
            if patient_id in common_participant_ids:
                try:
                    au_data = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)
                    au_data.columns = au_data.columns.str.strip()
                    au_data['Case'] = patient_id
                    all_au_features.append(au_data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        else:
            print(f"Skipping empty file: {file}")

if all_au_features:
    au_features_df = pd.concat(all_au_features, ignore_index=True)
    au_features_df['Case'] = au_features_df['Case'].astype(str)

    # Merge AU features with the labels
    combined_data = pd.merge(au_features_df, labels_df, on='Case', how='inner')

    # Drop unnecessary columns
    X_facial = combined_data.drop(columns=['frame', 'timestamp', 'confidence', 'success', 'Case', 'PHQ8'])
    y_facial = combined_data['PHQ8']
    participant_ids_facial = combined_data['Case'].values

    # Convert to numeric and handle missing values
    X_facial = X_facial.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Standardize the features
    scaler_facial = StandardScaler()
    X_facial_scaled = scaler_facial.fit_transform(X_facial)

    # Aggregate features by Participant_ID
    facial_df = pd.DataFrame(X_facial_scaled)
    facial_df['Participant_ID'] = participant_ids_facial
    facial_df['Facial_Label'] = y_facial.values

    facial_grouped = facial_df.groupby('Participant_ID').mean().reset_index()

    # Features and labels
    feature_columns_facial = facial_grouped.columns.drop(['Participant_ID', 'Facial_Label'])
    X_facial_grouped = facial_grouped[feature_columns_facial].values
    y_facial_grouped = facial_grouped['Facial_Label'].values
    participant_ids_facial_grouped = facial_grouped['Participant_ID'].values

    # Split data
    train_mask_facial = np.isin(participant_ids_facial_grouped, ids_train)
    test_mask_facial = np.isin(participant_ids_facial_grouped, ids_test)

    X_train_facial = X_facial_grouped[train_mask_facial]
    X_test_facial = X_facial_grouped[test_mask_facial]
    y_train_facial = y_facial_grouped[train_mask_facial]
    y_test_facial = y_facial_grouped[test_mask_facial]
    participant_ids_train_facial = participant_ids_facial_grouped[train_mask_facial]
    participant_ids_test_facial = participant_ids_facial_grouped[test_mask_facial]
else:
    raise ValueError("No AU features were loaded. Please check your AU files.")

# =========================================
# Step 6: Standardize Transcript Features
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_transcript)
X_test_scaled = scaler.transform(X_test_transcript)

# =========================================
# Step 7: Define and Train the Audio Model
# =========================================
# Define the model
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

# Train the model
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

history_seq = model_seq.fit(
    X_train_seq,
    y_train_seq,
    epochs=15,
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    verbose=1,
    callbacks=[lr_reduction]
)

# Evaluate the audio model
loss_seq, accuracy_seq = model_seq.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"Audio Model - Test Loss: {loss_seq:.4f}, Test Accuracy: {accuracy_seq:.4f}")

# =========================================
# Step 8: Train the Transcript Models
# =========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train_transcript)

# Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train_transcript)

# SVM Model
svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_transcript)

# =========================================
# Step 9: Train Models on Facial Features
# =========================================
# We'll use Random Forest, Naive Bayes, and SVM as in your code
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Random Forest Classifier
facial_rf_model = RandomForestClassifier(random_state=42)
facial_rf_model.fit(X_train_facial, y_train_facial)
facial_rf_predictions = facial_rf_model.predict(X_test_facial)

# Naive Bayes Model
facial_nb_model = GaussianNB()
facial_nb_model.fit(X_train_facial, y_train_facial)
facial_nb_predictions = facial_nb_model.predict(X_test_facial)

# SVM Model
facial_svm_model = SVC(probability=True, random_state=42)
facial_svm_model.fit(X_train_facial, y_train_facial)
facial_svm_predictions = facial_svm_model.predict(X_test_facial)

# =========================================
# Step 10: Get Predictions from Each Model
# =========================================
# Audio Model Predictions
audio_predictions_prob = model_seq.predict(X_test_seq)
audio_predictions_binary = (audio_predictions_prob >= 0.5).astype(int).flatten()

# Create a DataFrame with Participant IDs and predictions
audio_df = pd.DataFrame({
    'Participant_ID': participant_ids_test_seq,
    'Audio_Pred': audio_predictions_binary,
    'Audio_Label': y_test_seq
})

# Group by Participant_ID and take the mean prediction
audio_grouped = audio_df.groupby('Participant_ID').mean().reset_index()
audio_grouped['Audio_Pred'] = (audio_grouped['Audio_Pred'] >= 0.5).astype(int)
audio_grouped['Participant_ID'] = audio_grouped['Participant_ID'].astype(str)

# Transcript Model Predictions
rf_predictions = rf_model.predict(X_test_scaled)
nb_predictions = nb_model.predict(X_test_scaled)
svm_predictions = svm_model.predict(X_test_scaled)

# Create a DataFrame with Participant IDs and predictions
transcript_df = pd.DataFrame({
    'Participant_ID': participant_ids_test_transcript,
    'RF_Pred': rf_predictions.astype(int),
    'NB_Pred': nb_predictions.astype(int),
    'SVM_Pred': svm_predictions.astype(int),
    'Transcript_Label': y_test_transcript
})

# Take the majority vote for transcript predictions
transcript_grouped = transcript_df.groupby('Participant_ID').agg(lambda x: x.value_counts().index[0]).reset_index()
transcript_grouped['Participant_ID'] = transcript_grouped['Participant_ID'].astype(str)

# Facial Features Predictions
facial_predictions_df = pd.DataFrame({
    'Participant_ID': participant_ids_test_facial,
    'Facial_RF_Pred': facial_rf_predictions.astype(int),
    'Facial_NB_Pred': facial_nb_predictions.astype(int),
    'Facial_SVM_Pred': facial_svm_predictions.astype(int),
    'Facial_Label': y_test_facial
})

# Take the majority vote for facial predictions
facial_grouped_predictions = facial_predictions_df.groupby('Participant_ID').agg(lambda x: x.value_counts().index[0]).reset_index()
facial_grouped_predictions['Participant_ID'] = facial_grouped_predictions['Participant_ID'].astype(str)

# =========================================
# Step 11: Merge Predictions Based on Participant IDs
# =========================================
# Merge all DataFrames on Participant_ID
combined_df_dep = audio_grouped.merge(transcript_grouped, on='Participant_ID', how='inner')
combined_df_dep = combined_df_dep.merge(facial_grouped_predictions, on='Participant_ID', how='inner')

# Ensure labels are consistent
combined_df_dep['Label'] = combined_df_dep['Audio_Label'].astype(int)

# =========================================
# Step 12: Update the Combination Function
# =========================================
def determine_depression(facial_rf_pred, facial_nb_pred, facial_svm_pred,
                         audio_pred, rf_pred, nb_pred, svm_pred):
    """
    Determine if someone has depression based on weighted predictions.
    """
    # Define weights
    facial_weight = 4
    audio_weight = 3
    transcript_weight = 2
    method_weight_transcript = transcript_weight / 3  # Each transcript method gets an equal share
    method_weight_facial = facial_weight / 3  # Each facial method gets an equal share

    # Calculate the weighted sum
    total_score = (audio_pred * audio_weight) + \
                  (rf_pred * method_weight_transcript) + \
                  (nb_pred * method_weight_transcript) + \
                  (svm_pred * method_weight_transcript) + \
                  (facial_rf_pred * method_weight_facial) + \
                  (facial_nb_pred * method_weight_facial) + \
                  (facial_svm_pred * method_weight_facial)

    # Maximum possible score
    max_score = audio_weight + transcript_weight + facial_weight

    # Determine the final result based on the threshold (half of max_score)
    if total_score >= (max_score / 2):
        return 1  # Depression
    else:
        return 0  # No depression

# =========================================
# Step 13: Combine Predictions Using Updated Weights
# =========================================
# Apply the function to combine predictions
combined_df_dep['Final_Pred'] = combined_df_dep.apply(
    lambda row: determine_depression(
        row['Facial_RF_Pred'],
        row['Facial_NB_Pred'],
        row['Facial_SVM_Pred'],
        row['Audio_Pred'],
        row['RF_Pred'],
        row['NB_Pred'],
        row['SVM_Pred']
    ), axis=1
)

# =========================================
# Step 14: Evaluate the Combined Model
# =========================================
# True labels
y_true = combined_df_dep['Label'].astype(int)
final_predictions = combined_df_dep['Final_Pred'].astype(int)

# Accuracy
accuracy = accuracy_score(y_true, final_predictions)
print(f"Combined Model Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, final_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true, final_predictions)
print("Classification Report:")
print(class_report)

# Optional: Visualize ROC Curve
fpr, tpr, _ = roc_curve(y_true, final_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label=f'Combined Model ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Combined Model')
plt.legend(loc="lower right")
plt.show()
