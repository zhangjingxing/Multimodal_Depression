# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Fixing the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Load your dataset
data = pd.read_csv('Transcript_Feature_New.csv')

# Shuffle the data
data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Preparing the data
X = data.drop(['Case', 'PHQ8'], axis=1)
y = data['PHQ8']

# Standardizing features for models that require scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_seed)

# Initialize a dictionary to store results
results = {}

# Logistic Regression
logistic_model = LogisticRegression(random_state=random_seed)
logistic_model.fit(X_train, y_train)
results['Logistic Regression'] = accuracy_score(y_test, logistic_model.predict(X_test))

# Decision Trees
decision_tree_model = DecisionTreeClassifier(random_state=random_seed)
decision_tree_model.fit(X_train, y_train)
results['Decision Trees'] = accuracy_score(y_test, decision_tree_model.predict(X_test))

# Random Forest
random_forest_model = RandomForestClassifier(random_state=random_seed, n_estimators=100)
random_forest_model.fit(X_train, y_train)
results['Random Forest'] = accuracy_score(y_test, random_forest_model.predict(X_test))

# Gradient Boosting
gradient_boosting_model = GradientBoostingClassifier(random_state=random_seed)
gradient_boosting_model.fit(X_train, y_train)
results['Gradient Boosting'] = accuracy_score(y_test, gradient_boosting_model.predict(X_test))

# Support Vector Machines
svm_model = SVC(kernel='linear', probability=True, random_state=random_seed)
svm_model.fit(X_train, y_train)
results['Support Vector Machines'] = accuracy_score(y_test, svm_model.predict(X_test))

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
results['K-Nearest Neighbors'] = accuracy_score(y_test, knn_model.predict(X_test))

# Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
results['Naive Bayes'] = accuracy_score(y_test, naive_bayes_model.predict(X_test))

# Linear Discriminant Analysis
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
results['Linear Discriminant Analysis'] = accuracy_score(y_test, lda_model.predict(X_test))

# Display results
for model, accuracy in results.items():
    print(f"{model}: {accuracy:.4f}")

# Plotting Data & Result
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# New Section: Visualization

# Binary PHQ8 Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Binary PHQ8 Distribution')
plt.xlabel('PHQ8')
plt.ylabel('Count')
plt.show()

# Random Forest Feature Importance
plt.figure(figsize=(10, 8))
importance = random_forest_model.feature_importances_
features = X.columns
sns.barplot(x=importance, y=features)
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Random Forest Tree Structure
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(random_forest_model.estimators_[0], filled=True, feature_names=X.columns, class_names=['0', '1'], fontsize=10)
plt.title('Random Forest Tree Structure')
plt.show()

# Naive Bayes Feature Contribution
plt.figure(figsize=(10, 8))
mean_diff = naive_bayes_model.theta_[1] - naive_bayes_model.theta_[0]
sns.barplot(x=mean_diff, y=features)
plt.title('Naive Bayes Feature Contribution')
plt.xlabel('Mean Difference (Class 1 - Class 0)')
plt.ylabel('Features')
plt.show()

# Confusion Matrices
for model_name, model in [('SVM', svm_model),
                          ('Random Forest', random_forest_model),
                          ('Naive Bayes', naive_bayes_model)]:
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap='viridis')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# ROC Curve
plt.figure(figsize=(10, 8))
for model_name, model in [('SVM', svm_model),
                          ('Random Forest', random_forest_model),
                          ('Naive Bayes', naive_bayes_model)]:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)  # For SVM
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for model_name, model in [('SVM', svm_model),
                          ('Random Forest', random_forest_model),
                          ('Naive Bayes', naive_bayes_model)]:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)  # For SVM
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=model_name)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right')
plt.show()
