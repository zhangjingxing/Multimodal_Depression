# Multimodal_Depression
A project using CNNs and RNNs to detect depression from multimodal data: speech recordings (acoustic features), transcripts (linguistic analysis), and facial features (action units for non-verbal cues). Integrates diverse data sources for accurate detection of depression and to gain the underlying insights behind mental health issue detection.

The dataset used in this project is from the DCAPS-WOZ Database, which contains clinical interviews designed to support the diagnosis of psychological distress conditions such as anxiety, depression, and post-traumatic stress disorder (PTSD). The data were collected through Wizard-of-Oz interviews conducted by an animated virtual interviewer, Ellie, controlled by a human interviewer in another room. Includes 189 sessions of interactions ranging between 7-33 minutes (average ~16 minutes). Contains audio and video recordings, transcripts of interactions, and annotations for verbal and non-verbal features.

Due to consent constraints, the dataset is NOT provided here, and available only to academics and non-profit researchers. Interested parties must complete and sign a request form using an academic email address to gain access through this website: [DCAPS-WOZ Database](https://dcapswoz.ict.usc.edu/). You should expect to receive the permission in several days.

## Audio Preprocessing
The raw audio recordings include both the interviewer’s and the participant’s voices. To ensure the data focuses solely on the participant, we preprocess the audio by cropping out segments where the interviewer speaks. Once the participant's speech is isolated, we extract acoustic features using techniques such as Mel-Frequency Cepstral Coefficients (MFCC). MFCC effectively captures the spectral and temporal characteristics of the audio, making it suitable for identifying speech patterns associated with depression.

## Transcript Preprocessing
The dataset includes interview scripts documenting interactions between the interviewer and participants. To preprocess this text data, we leverage ChatGPT to identify key linguistic features indicative of depression. ChatGPT generates a list of these features and assigns a score (ranging from 1 to 5) for each participant based on their transcript. These scores are then used as input for training the machine learning model. Here is the prompt we used to make a custom ChatGPT model [Transcript Processing Prompt](https://docs.google.com/document/d/1-fOb1O6eGVn1u-EGGAuLoVHyopnJMHrD4SBmxVuP0bc/edit?usp=sharing) The performance of ChatGPT might differ now since it already released a newer version, so the older version's performance might be slightly different or even getting worse. 

## Facial Feature Preprocessing
The dataset contains comprehensive facial feature data, including pose, gaze, and Action Units (AUs). For our analysis, we focus specifically on Action Units, which represent facial muscle movements and are intuitive and reliable indicators of emotional states. AUs are particularly effective in detecting emotions such as sadness or lack of expression, both of which are significant in identifying depression. These features are directly used to train the model, capturing the non-verbal cues of participants during the interviews.



