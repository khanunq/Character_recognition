import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Get the data and labels
data = data_dict['data']
labels = data_dict['labels']


# Remove the elements at indices 1729 and 1733
del data[1729]
del data[1733]  # Note: After removing the element at index 1729, the index 1733 will shift to 1732
del data[1732]
del labels[1729]
del labels[1733]
del labels[1732]

# Create a new dictionary with the updated data and labels
data_dict = {'data': data, 'labels': labels}

# Reshape the data elements to a consistent shape
target_shape = (42,)  # Assuming the desired shape is (42,)
preprocessed_data = []
for data_element in data_dict['data']:
    preprocessed_data.append(np.array(data_element).reshape(target_shape))

# Convert preprocessed data and labels to NumPy arrays
data = np.asarray(preprocessed_data)
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model on the test set
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a pickle file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()