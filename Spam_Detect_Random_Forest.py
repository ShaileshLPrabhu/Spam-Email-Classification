import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# Classification
classifier = RandomForestClassifier(n_estimators=10, criterion="gini")
classifier.fit(x_train_vectorized, y_train)
y_pred = classifier.predict(x_test_vectorized)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy * 100))

# Creating the Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred)

# Display the Confusion Matrix using the Seaborn heatmap
sns.heatmap(cm_rf, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Random Forest')
plt.show()