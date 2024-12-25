# =========================================
# Combined Code for Predicting Depression using Transcript and Audio Models
# =========================================

# Step 0: Import Necessary Libraries and Define Functions
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Define MFCC Extraction Function
def compute_mfcc_sequences(file_path, sr=16000, n_mfcc=13, hop_length=512,
                           frame_size=100, sequence_length=10, noise_factor=0.005):

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

# Define Majority Voting Function
def majority_vote(predictions):
    """
    Performs majority voting on a list of predictions.

    Parameters:
    - predictions (array-like): Array of predictions (e.g., [0, 1, 1])

    Returns:
    - int: The majority class label.
    """
    counts = np.bincount(predictions)
    return np.argmax(counts)


# Step 2: Load and Preprocess New Transcript Data

data = {
    'Case': ['Angie', 'Jeremy', 'Jonathan', 'Richard', 'VIncent', 'Winnie'],
    'Sentiment Consistency': [4, 3, 5, 4, 3, 4],
    'Emotional Variability': [3, 3, 4, 4, 3, 3],
    'Self-Reflection and Insight': [4, 3, 5, 4, 4, 4],
    'Social Interaction Indicators': [4, 4, 4, 3, 4, 3],
    'Repetition and Ruminative Language': [4, 3, 4, 3, 3, 4],
    'Energy and Motivation Levels': [4, 3, 4, 3, 3, 4],
    'Response Length and Depth': [4, 3, 4, 4, 3, 4],
    'PHQ8': [0, 0, 0, 0, 1, 0]
}

# Create DataFrame
new_transcript_df = pd.DataFrame(data)

print("New Transcript Data:")
print(new_transcript_df)

# Step 3: Load and Preprocess Corresponding Audio Data
new_audio_directory = 'drive/MyDrive/group data ml/MP3'  # Adjust if different

# Initialize lists to store audio sequences and corresponding participant IDs
audio_sequences = []
audio_labels = []  # Assuming labels are known or retrieved from transcript data
participant_ids_audio = []

# Iterate over each row in the transcript DataFrame
for index, row in new_transcript_df.iterrows():
    participant_id = row['Case']
    file_name = f"{participant_id}.wav"
    file_path = os.path.join(new_audio_directory, file_name)

    if os.path.exists(file_path):
        sequences = compute_mfcc_sequences(
            file_path,
            sr=16000,
            n_mfcc=13,
            hop_length=512,
            frame_size=100,
            sequence_length=10
        )
        if sequences is not None:
            audio_sequences.append(sequences)
            audio_labels.extend([row['PHQ8']] * sequences.shape[0])
            participant_ids_audio.extend([participant_id] * sequences.shape[0])
        else:
            print(f"No sequences extracted from {file_name}")
    else:
        print(f"Audio file not found: {file_name}")

# Convert lists to NumPy arrays
if audio_sequences:
    audio_sequences = np.vstack(audio_sequences)
    audio_labels = np.array(audio_labels)
    participant_ids_audio = np.array(participant_ids_audio)
    print(f"Total audio sequences: {audio_sequences.shape[0]}")
else:
    print("No audio sequences were extracted. Please check your audio files.")

# Step 4: Make Predictions Using Transcript and Audio Models

# Transcript Predictions
transcript_features = [
    'Sentiment Consistency',
    'Emotional Variability',
    'Self-Reflection and Insight',
    'Social Interaction Indicators',
    'Repetition and Ruminative Language',
    'Energy and Motivation Levels',
    'Response Length and Depth'
]

X_new_transcript = new_transcript_df[transcript_features].values
y_new_transcript = new_transcript_df['PHQ8'].values  # Not used for prediction

# Scale the new transcript data using the scaler fitted during training
X_new_transcript_scaled = scaler.transform(X_new_transcript)

# Transcript Model Predictions
transcript_rf_pred = rf_model.predict(X_new_transcript_scaled)
transcript_nb_pred = nb_model.predict(X_new_transcript_scaled)
transcript_svm_pred = svm_model.predict(X_new_transcript_scaled)

# Combine Transcript Predictions
transcript_predictions = np.vstack([
    transcript_rf_pred,
    transcript_nb_pred,
    transcript_svm_pred
]).T

# Apply majority voting for Transcript Predictions
transcript_majority_pred = np.apply_along_axis(majority_vote, axis=1, arr=transcript_predictions)

# Add Transcript Predictions to DataFrame
new_transcript_df = new_transcript_df.copy()
new_transcript_df['Transcript_Pred'] = transcript_majority_pred

# Audio Model Predictions
if audio_sequences.size == 0:
    print("No audio sequences available for prediction.")
    # Assign default predictions (e.g., Not Depressed)
    new_transcript_df['Audio_Pred'] = 0
else:
    # Predict probabilities using the Audio model
    audio_pred_prob = model_seq.predict(audio_sequences)

    # Convert probabilities to binary predictions using a threshold of 0.5
    audio_pred_binary = (audio_pred_prob >= 0.5).astype(int).flatten()

    # Create a DataFrame with Participant IDs and predictions
    audio_df = pd.DataFrame({
        'Participant_ID': participant_ids_audio,
        'Audio_Pred': audio_pred_binary,
        'Audio_Label': audio_labels
    })

    # Group by Participant_ID and take the majority vote for each participant
    audio_grouped = audio_df.groupby('Participant_ID')['Audio_Pred'].apply(majority_vote).reset_index()
    audio_grouped.rename(columns={'Audio_Pred': 'Audio_Pred'}, inplace=True)

    # Ensure 'Case' and 'Participant_ID' are strings for merging
    new_transcript_df['Case'] = new_transcript_df['Case'].astype(str)
    audio_grouped['Participant_ID'] = audio_grouped['Participant_ID'].astype(str)

    # Merge Audio Predictions with Transcript Data
    final_predictions_df = new_transcript_df.merge(
        audio_grouped,
        left_on='Case',
        right_on='Participant_ID',
        how='left'
    )

    # Handle participants with no audio predictions by assigning a default value
    final_predictions_df['Audio_Pred'].fillna(0, inplace=True)  # Assuming default to 'Not Depressed'
    final_predictions_df['Audio_Pred'] = final_predictions_df['Audio_Pred'].astype(int)

# Combine Transcript and Audio Predictions via Majority Voting
if audio_sequences.size == 0:
    # If no audio predictions, final prediction is based on Transcript
    final_predictions_df['Final_Pred'] = final_predictions_df['Transcript_Pred']
else:
    combined_predictions = np.vstack([
        final_predictions_df['Transcript_Pred'].values,
        final_predictions_df['Audio_Pred'].values
    ]).T

    # Apply majority voting
    final_predictions = np.apply_along_axis(majority_vote, axis=1, arr=combined_predictions)

    # Add Final Predictions to DataFrame
    final_predictions_df['Final_Pred'] = final_predictions

# Step 5: Visualize the Results

# True Labels and Predicted Labels
if audio_sequences.size == 0:
    true_labels = final_predictions_df['PHQ8'].values
    predicted_labels = final_predictions_df['Final_Pred'].values
else:
    true_labels = final_predictions_df['PHQ8'].values
    predicted_labels = final_predictions_df['Final_Pred'].values

# 1. Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Depressed', 'Depressed'])

plt.figure(figsize=(6,6))
disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False, values_format='d')
plt.title('Confusion Matrix', fontsize=16)
plt.show()

# 2. Prediction Distribution Bar Plot
plot_df = final_predictions_df.copy()
plot_df['Final_Pred_Label'] = plot_df['Final_Pred'].map({0: 'Not Depressed', 1: 'Depressed'})

plt.figure(figsize=(8,6))
sns.countplot(x='Final_Pred_Label', data=plot_df, palette='Set2')
plt.title('Final Prediction Distribution', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 4. Feature Importances (Transcript Random Forest)
rf_importances = rf_model.feature_importances_
features = transcript_features
rf_feature_importances = pd.Series(rf_importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=rf_feature_importances.values, y=rf_feature_importances.index, palette='viridis')
plt.title('Transcript Random Forest Feature Importances', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.show()

# 5. Classification Report (Optional)
class_report = classification_report(true_labels, predicted_labels, target_names=['Not Depressed', 'Depressed'])
print("Classification Report:")
print(class_report)

# Step 6: Print Final Predictions
print("Final Predictions DataFrame:")
print(final_predictions_df[['Case', 'Transcript_Pred', 'Audio_Pred', 'Final_Pred']])
