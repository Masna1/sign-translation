# import pickle
# from sklearn.svm import SVC  
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# data_dict = pickle.load(open('./data1.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = SVC()  # You can specify parameters like kernel='linear', C=1.0, etc.

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)
# print(score)
# print('{}% of samples were classified correctly!'.format(score * 100))

# f = open('model_svm.p', 'wb')  # Saving the SVM model to a different file
# pickle.dump({'model': model}, f)
# f.close()




# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data1.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()



import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load data
data_dict = pickle.load(open('./data1_processed.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Get unique labels and their counts
unique_labels, label_counts = np.unique(labels, return_counts=True)
# Filter labels that have at least two instances
valid_labels = unique_labels[label_counts >= 2]
valid_mask = np.isin(labels, valid_labels)
data = data[valid_mask]
labels = labels[valid_mask]


# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.9, shuffle=True, stratify=labels)
input_layer_size = x_train.shape[1]
print(input_layer_size)
output_layer_size = len(np.unique(y_train))
print(output_layer_size )

# Initialize and train an ANN model
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000)  
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)


# Calculate evaluation metrics
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

# Print evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

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

# Save the trained model
with open('model_ann.p', 'wb') as f:
    pickle.dump({'model': model}, f)
