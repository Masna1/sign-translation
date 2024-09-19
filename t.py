
# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict_words = pickle.load(open('./data2.pickle', 'rb'))

# data = np.asarray(data_dict_words['data'])
# labels = np.asarray(data_dict_words['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model1 = RandomForestClassifier()

# model1.fit(x_train, y_train)

# y_predict = model1.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model1': model1}, f)
# f.close()




# import pickle
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load data
# data_dict_words = pickle.load(open('./data2.pickle', 'rb'))
# data = np.asarray(data_dict_words['data'])
# labels = np.asarray(data_dict_words['labels'])

# # Split data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Choose a different classifier (SVM)
# model2 = SVC()

# # Train the model
# model2.fit(x_train, y_train)

# # Make predictions on the test set
# y_predict = model2.predict(x_test)

# # Evaluate the accuracy
# score = accuracy_score(y_predict, y_test)

# # Print the accuracy
# print('{}% of samples were classified correctly using SVM!'.format(score * 100))

# # Save the trained SVM model
# f = open('model_svm.p', 'wb')
# pickle.dump({'model2': model2}, f)
# f.close()

import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict_words = pickle.load(open('./data2.pickle', 'rb'))

data = np.asarray(data_dict_words['data'])
labels = np.asarray(data_dict_words['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.9, shuffle=True, stratify=labels)
input_layer_size = x_train.shape[1]
print(input_layer_size)
output_layer_size = len(np.unique(y_train))
print(output_layer_size )

# Define and train the MLP classifier
model1 = MLPClassifier(hidden_layer_sizes=(48,32), max_iter=5000)

model1.fit(x_train, y_train)

# Predict on the test set
y_predict = model1.predict(x_test)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate predictions
y_predict = model1.predict(x_test)

# Calculate evaluation metrics
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

# Print evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model1': model1}, f)


