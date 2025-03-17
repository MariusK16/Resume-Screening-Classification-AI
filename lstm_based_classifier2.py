import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
import gensim.downloader as api
import matplotlib.pyplot as plt
# -------------------------------
# 1. Load and Prepare the Dataset
# -------------------------------
# Load the preprocessed dataset with a "cleaned_resume" and "Category" column
df = pd.read_csv("cleaned_resume_processed.csv")
df['cleaned_resume'] = df['cleaned_resume'].astype(str)

# Extract texts and target labels
texts = df['cleaned_resume'].tolist()
labels = df['Category'].tolist()

# Encode target labels to integers
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
y = to_categorical(encoded_labels, num_classes=num_classes)

# -------------------------------
# 2. Tokenize and Pad the Text Data
# -------------------------------
max_words = 5000           # Maximum vocabulary size
max_sequence_length = 200  # Maximum length of each resume sequence

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_sequence_length)

# -------------------------------
# Split data into train and test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# -------------------------------
# 3. Prepare the Pre-trained Embedding Matrix
# -------------------------------
# Load pre-trained GloVe embeddings via Gensim (50-dimensional)
embedding_model = api.load("glove-wiki-gigaword-50")
embedding_dim = embedding_model.vector_size

# Build an embedding matrix for our vocabulary
word_index = tokenizer.word_index
vocab_size = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if i < vocab_size:
        if word in embedding_model.key_to_index:
            embedding_matrix[i] = embedding_model[word]
        else:
            # For words not in the pre-trained model, assign a random vector
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# -------------------------------
# 4. Build the LSTM Model
# -------------------------------
def build_model(hp):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=True  # Set to True to fine-tune embeddings if desired
        )
    )

    # Tune the number of LSTM units between 32 and 128 (step size 32)
    lstm_units = hp.Int('lstm_units', min_value=128, max_value=256, step=32)
    # Tune dropout rates
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.3, step=0.1)
    recurrent_dropout_rate = hp.Float('recurrent_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    model.add(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# -------------------------------
# 5. Train the Model
# -------------------------------


# Create a tuner instance; adjust max_trials as needed
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='resume_classifier_tuning'
)

# Use 20% of the training data as validation data during tuning.
tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print("LSTM units:", best_hps.get('lstm_units'))
print("Dropout rate:", best_hps.get('dropout_rate'))
print("Recurrent dropout rate:", best_hps.get('recurrent_dropout_rate'))

# Optionally, build the model with the best hyperparameters and retrain
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)
best_model.save("best_lstm_resume_classifier.h5")

# plot values vs accuracy/loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# linr plot for training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# heatmap showing correlation between hyperparameters and performance
tuner.results_summary()



# # Optionally, save the model
# model.save("lstm_resume_classifier.h5")

print("LSTM model training complete!")


# # If your model is already trained and saved as "lstm_resume_classifier.h5"
# # Otherwise, use the trained model from your session.
# model = load_model("lstm_resume_classifier.h5")

# # Evaluate the model using model.evaluate()
# loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
# print("\nTest Loss: {:.4f}, Test Accuracy: {:.4f}".format(loss, accuracy))

# # Generate predictions and compute detailed metrics
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # Print a classification report (includes precision, recall, f1-score)
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=le.classes_))

# # Alternatively, you can calculate accuracy, precision, and recall individually:
# acc = accuracy_score(y_true, y_pred)
# prec = precision_score(y_true, y_pred, average='weighted')
# rec = recall_score(y_true, y_pred, average='weighted')
# print("Accuracy: {:.4f}".format(acc))
# print("Precision: {:.4f}".format(prec))
# print("Recall: {:.4f}".format(rec))
