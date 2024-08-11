import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Read the data from CSV file into a DataFrame
df = pd.read_csv('email.csv')

# checking for missing values
# print(df.isnull().sum()) - no missing values found

# Removing Duplicates values

# print(df.duplicated().sum())- 415 duplicates
df.drop_duplicates(inplace=True)

# Divide the data into training and test
x_train, x_test, y_train, y_test = train_test_split(df["Message"], df["Category"], test_size=0.2, random_state=42)

# Vectorize the data
cv = CountVectorizer()
x_train_vectorized = cv.fit_transform(x_train)
x_test_vectorized = cv.transform(x_test)

# classification

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train_vectorized, y_train)
y_pred = classifier.predict(x_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy * 100))

# Creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the Confusion Matrix using Seaborn heatmap

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Decision Tree')
plt.show()