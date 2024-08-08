import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Read the data from CSV file into a DataFrame
df = pd.read_csv('email.csv')

# Removing duplicate values
df.drop_duplicates(inplace=True)

# Divide the data into training and test
x_train, x_test, y_train, y_test = train_test_split(df["Message"], df["Category"], test_size=0.2, random_state=42)

# Vectorize the data
cv = CountVectorizer()
x_train_vectorized = cv.fit_transform(x_train)
x_test_vectorized = cv.transform(x_test)

# SVM Classifier
sv = svm.SVC()
sv.fit(x_train_vectorized, y_train)
y_pred_svm = sv.predict(x_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_svm)
print('Accuracy: %.2f' % (accuracy * 100))

# Creating the Confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Display the Confusion Matrix using Seaborn heatmap
sns.heatmap(cm_svm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - SVM')
plt.show()
