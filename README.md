# Multimodal_Depression
A project using CNNs and RNNs to detect depression from multimodal data: speech recordings (acoustic features), transcripts (linguistic analysis), and facial features (action units for non-verbal cues). Integrates diverse data sources for accurate detection of depression and to gain the underlying insights behind mental health issue detection.

The dataset used in this project is from the DCAPS-WOZ Database, which contains clinical interviews designed to support the diagnosis of psychological distress conditions such as anxiety, depression, and post-traumatic stress disorder (PTSD). The data were collected through Wizard-of-Oz interviews conducted by an animated virtual interviewer, Ellie, controlled by a human interviewer in another room. Includes 189 sessions of interactions ranging between 7-33 minutes (average ~16 minutes). Contains audio and video recordings, transcripts of interactions, and annotations for verbal and non-verbal features.

Due to consent constraints, the dataset is NOT provided here, and available only to academics and non-profit researchers. Interested parties must complete and sign a request form using an academic email address to gain access through this website: [DCAPS-WOZ Database](https://dcapswoz.ict.usc.edu/). You should expect to receive the permission in several days.

## Audio Preprocessing
The raw audio recordings include both the interviewer’s and the participant’s voices. To ensure the data focuses solely on the participant, we preprocess the audio by cropping out segments where the interviewer speaks. Once the participant's speech is isolated, we extract acoustic features using techniques such as Mel-Frequency Cepstral Coefficients (MFCC). For more information on how to extract Mel-Frequency Cepstral Coefficients (MFCCs), refer to the [Librosa MFCC documentation](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html?utm_source=chatgpt.com). MFCC effectively captures the spectral and temporal characteristics of the audio, making it suitable for identifying speech patterns associated with depression.

## Transcript Preprocessing
The dataset includes interview scripts documenting interactions between the interviewer and participants. To preprocess this text data, we leverage ChatGPT to identify key linguistic features indicative of depression. ChatGPT generates a list of these features and assigns a score (ranging from 1 to 5) for each participant based on their transcript. These scores are then used as input for training the machine learning model. Here is the prompt we used to make a custom ChatGPT model [Transcript Processing Prompt](https://docs.google.com/document/d/1-fOb1O6eGVn1u-EGGAuLoVHyopnJMHrD4SBmxVuP0bc/edit?usp=sharing) The performance of ChatGPT might differ now since it already released a newer version, so the older version's performance might be slightly different or even getting worse. 

## Facial Feature Preprocessing
The dataset contains comprehensive facial feature data, including pose, gaze, and Action Units (AUs). For our analysis, we focus specifically on Action Units, which represent facial muscle movements and are intuitive and reliable indicators of emotional states. AUs are particularly effective in detecting emotions such as sadness or lack of expression, both of which are significant in identifying depression. These features are already provided, so we can directly use them to train the model, capturing the non-verbal cues of participants.

## Model Architectures
### Transcript Model
To analyze the linguistic features of transcripts, we trained multiple models, including SVM (with a linear kernel), Random Forest, Naive Bayes, Logistic Regression, and Linear Discriminant Analysis (LDA). Among these, SVM with a linear kernel performed the best, achieving the highest accuracy and indicating that the dataset is primarily linearly separable.

Random Forest, though slightly less accurate, was able to capture minor non-linear patterns, suggesting some complexity or noise in the data. Naive Bayes, which assumes feature independence, performed similarly to Random Forest, reinforcing the dataset’s general alignment with the target variable while not fully addressing non-linear interactions. Logistic Regression and LDA, which also rely on linear decision boundaries, had lower accuracy, further confirming the dataset’s linear nature with subtle non-linear nuances.

### Audio Model
For audio data, we designed a hybrid model combining CNNs and LSTMs to capture both spatial and temporal patterns. The input data consisted of Mel-Frequency Cepstral Coefficients (MFCCs) extracted from participants’ speech recordings. These MFCCs were framed and sequenced to preserve temporal coherence.

The CNN component used TimeDistributed Conv2D layers to extract spatial features from MFCC frames, followed by batch normalization and max-pooling for stable learning and feature reduction. The processed outputs were then fed into stacked LSTM layers to capture temporal dependencies across sequences. A final dense layer with a sigmoid activation function provided binary classifications: 1 for depression and 0 for no depression.

The dataset was split into an 80:20 ratio for training and testing, ensuring balanced label distribution. The model achieved a validation accuracy of 83% and a validation loss of 38%, showcasing its effectiveness in detecting depression markers from speech.

### Face Model
First, we collected AU feature files for each participant. These files were linked to each participant by their unique ID, which was extracted from the filenames. The AU features were then merged with transcript features to match each participant's data with their PHQ8 score, which is used as the target label.

The data was split into 80% for training and 20% for testing. To improve model performance, we standardized the features so they all had the same scale.

We tested three machine learning models: Naive Bayes, Random Forest, and SVM. Naive Bayes didn't perform well, especially for depression cases, due to class imbalance and its assumptions about the data. Random Forest performed the best, with a recall of 63% and precision of 88% for detecting depression. It was effective because it handles imbalanced data well and can capture complex patterns. SVM was also tested but didn't perform better than Random Forest.


