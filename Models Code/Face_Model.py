import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Path to AU feature files
au_files_path = 'drive/MyDrive/ML data/practice/AUs'  # Replace with your actual AU files folder
transcript_features_file = 'drive/MyDrive/ML data/Transcript_Feature.csv'  # Path to your transcript features file

# Load the transcript features (which already contains PHQ8 as the last column)
transcript_features = pd.read_csv(transcript_features_file)

# Extract the PHQ8 label (last column) and 'Case' (patient ID)
labels = transcript_features[['Case', 'PHQ8']]

# Ensure the 'Case' column is the same type in both DataFrames (convert to int64)
transcript_features['Casef'] = transcript_features['Case'].astype(int)

# Initialize an empty list to hold AU features for all patients
all_au_features = []

# Loop through AU files and load them
for file in os.listdir(au_files_path):
    if file.endswith('.txt'):  # Ensure it's an AU feature file
        file_path = os.path.join(au_files_path, file)

        # Check if the file is empty
        if os.path.getsize(file_path) > 0:
            patient_id = file.split('_')[0]  # Extract patient ID from filename (e.g., '397' from '397_CLNF_features.txt')

            try:
                # Load AU features (assuming the file has columns like frame, timestamp, confidence, etc.)
                au_data = pd.read_csv(file_path, delimiter=',')  # Adjust delimiter if needed
                au_data['Case'] = int(patient_id)  # Add Case as patient_id to the features

                # Ensure the 'Case' column is the same type in both DataFrames (convert to int64)
                au_data['Case'] = au_data['Case'].astype(int)

                # Append to the list of AU features
                all_au_features.append(au_data)
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"Skipping empty file: {file}")

# Concatenate all AU feature data into one DataFrame
if all_au_features:  # Ensure there's data to concatenate
    au_features_df = pd.concat(all_au_features, ignore_index=True)

    # Merge AU features with the transcript features (using 'Case' as patient_id)
    combined_data = pd.merge(au_features_df, transcript_features, on='Case', how='inner')

    # Now we have the combined data with AU features and PHQ8 labels
    # Drop 'Case' as it is not needed for modeling
    X = combined_data.drop(columns=['Case', 'PHQ8'])  # Features (drop Case and PHQ8)
    y = combined_data['PHQ8']  # Labels (PHQ8 score or binary label)

    # Split into training and test data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features (scaling for better performance in some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------- Naive Bayes Model -----------
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    y_pred_nb = nb_model.predict(X_test_scaled)
    print("Naive Bayes Model Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Classification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

    # ----------- Random Forest Classifier -----------
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("Random Forest Model Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

    # ----------- Support Vector Machine (SVM) -----------
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("Support Vector Machine Model Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm))

    # Optionally: If you want to compare all models in terms of accuracy:
    print("\nModel Comparison (Accuracy):")
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    #print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

else:
    print("No AU features were loaded. Please check your AU files.")