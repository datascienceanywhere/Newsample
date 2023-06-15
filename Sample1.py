import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the engagement combinations
df['sequence_tokens'] = df['sequence'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Prepare input sequences
max_sequence_length = max(df['sequence_tokens'].apply(len))
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(df['sequence_tokens'],
                                                                maxlen=max_sequence_length,
                                                                padding='post')
input_sequences = tf.convert_to_tensor(padded_sequences)

# Prepare numerical features
numerical_features = ['num_feature1', 'num_feature2']
numerical_data = df[numerical_features].values
input_numerical = tf.convert_to_tensor(numerical_data, dtype=tf.float32)

# Prepare target variable
target_variable = tf.convert_to_tensor(df['expected_rx_writing'].values, dtype=tf.float32)

# CF-hybridbert4rec component
user_input = Input(shape=(max_sequence_length,), dtype=tf.int32)
item_input = Input(shape=(max_sequence_length,), dtype=tf.int32)

bert_model = TFBertModel.from_pretrained('bert-base-uncased')
user_embedding = bert_model(user_input)[0][:, 0, :]  # Use the [CLS] token embedding
item_embedding = bert_model(item_input)[0][:, 0, :]  # Use the [CLS] token embedding

# CBF-hybridbert4rec component
numerical_input = Input(shape=(len(numerical_features),), dtype=tf.float32)

# Shared layers between CF-hybridbert4rec and CBF-hybridbert4rec
shared_dense = Dense(64, activation='relu')

# CF-hybridbert4rec processing
user_output = shared_dense(user_embedding)
item_output = shared_dense(item_embedding)

# CBF-hybridbert4rec processing
numerical_output = shared_dense(numerical_input)

# Concatenate CF-hybridbert4rec and CBF-hybridbert4rec outputs
concatenated = Concatenate()([user_output, item_output, numerical_output])

# Final prediction
output = Dense(1)(concatenated)

# Define HybridBERT4Rec model
model = Model(inputs=[user_input, item_input, numerical_input], outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit([input_sequences, input_sequences, input_numerical], target_variable, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict([input_sequences, input_sequences, input_numerical])
In this updated code, both CF-hybridbert4rec and CBF-hybridbert4rec components are defined.
