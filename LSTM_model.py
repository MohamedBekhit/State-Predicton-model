# Internal Imports
from utils import load_preprocess_data
from utils import shuffle_split_dataset, extract_categories, format_data

# External Imports
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keras import models
from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.optimizers import Adam
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle


def predict_with_uncertainty(f, x, no_classes, n_iter=100):
    result = np.zeros((n_iter,) + (x.shape[0], no_classes) )

    for i in range(n_iter):
        result[i,:, :] = f((x, 1))[0]

    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)
    return prediction, uncertainty


# Load Data
print("Loading and preprocessing Data...\n")
clean_data = load_preprocess_data()

# Preprocess Data
X = clean_data.drop('state', axis=1)
y = pd.DataFrame(clean_data['state'].values)
y.columns = ['state']
X = X.iloc[:500000]
y = y.iloc[:500000]
X = format_data(X)
del clean_data

# Extract categories
cats = extract_categories(y['state'].values)
cats.sort()
print(type(cats))
NUM_CATS = len(cats)
print("categories: ", cats)
print("number of categories: ", NUM_CATS)

# Scale Data
min_max_scaler = MinMaxScaler()
nsamples, nx, ny = X.shape
d2_X = X.reshape((nsamples,nx*ny))
d2_X_scaled = min_max_scaler.fit_transform(d2_X)
X_scaled = d2_X_scaled.reshape(nsamples, nx, ny)

del d2_X
del d2_X_scaled

# Shuffle and split Data
y = np.array(y['state'].to_numpy(dtype=np.float32))
X_train, y_train, X_test, y_test = shuffle_split_dataset(X_scaled, y)
train_cat = extract_categories(y_train)
test_cat = extract_categories(y_test)
print("\nX_train dims: ", X_train.shape)
print("X_test dims: ", X_test.shape)
print("y_train dims:", y_train.shape)
print("y_test dims:", y_test.shape)
print('_'*100)

# One Hot encode targets
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape)
print(y_test.shape)
enc = OneHotEncoder(categories='auto')
y_train_enc = enc.fit_transform(y_train).toarray()
y_test_enc = enc.fit_transform(y_test).toarray()
print(y_train_enc.shape)
print("_"*100)


# Build LSTM:
NUM_COLS = X.shape[1:]
print('NUM_COLS', NUM_COLS)
epochs = 200


model = models.Sequential()
model.add(CuDNNLSTM(300, input_shape=(X.shape[1:]), return_sequences=True))
model.add(Dropout(0.5))
model.add(CuDNNLSTM(300))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CATS, activation='softmax'))
model.add(Dropout(0.5))
f = K.function([model.layers[0].input, K.learning_phase()],
               [model.layers[-1].output])
model.summary()

early_stopping_monitor = EarlyStopping(patience=10)
model.compile(Adam(lr=.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train_enc, validation_split=0.2, batch_size=100,
                    callbacks=[early_stopping_monitor], epochs=epochs, shuffle=False, verbose=2)

# Evaluate model
eval = model.evaluate(X_test, y_test_enc, batch_size=100, verbose=2)
print("Accuracy = ", eval[1], "\tLoss = ", eval[0])
y_pred, uncertainty = predict_with_uncertainty(f=f, x=X_test, no_classes=NUM_CATS
                                               , n_iter=100)
print(y_pred.shape)
print(uncertainty)

save_model = open('model.pickle', 'wb')
pickle.dump(model, save_model)
save_model.close()

# Loss Visualization
plt.figure()
plt.plot(history.history['loss'])
plt.show()
#
# plt.figure()
# plt.scatter(range(1000), y_pred[:1000, :], c='r')
# plt.scatter(range(1000), y_test[:1000, :], c='g')
# plt.show()
