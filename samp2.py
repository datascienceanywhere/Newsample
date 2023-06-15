
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

class CfHybridBERT4Rec(tf.keras.Model):
    def __init__(self, num_users, num_items, num_numerical_features, fusion_dim, output_dim):
        super(CfHybridBERT4Rec, self).__init__()

        # Collaborative Filtering component
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_dim)
        self.fc_cf = tf.keras.layers.Dense(fusion_dim)

        # BERT-based Textual Encoding component
        self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        self.fc_bert = tf.keras.layers.Dense(fusion_dim)

        # Fusion and Prediction layer
        self.fc_fusion = tf.keras.layers.Dense(output_dim)

    def call(self, user_ids, item_ids, numerical_features, text_inputs):
        # Collaborative Filtering
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        cf_inputs = tf.concat([user_emb, item_emb, numerical_features], axis=1)
        cf_outputs = self.fc_cf(cf_inputs)

        # BERT-based Textual Encoding
        bert_inputs = self.bert_model(text_inputs)[1]  # [CLS] token output
        bert_outputs = self.fc_bert(bert_inputs)

        # Fusion and Prediction
        fusion_inputs = tf.concat([cf_outputs, bert_outputs], axis=1)
        predictions = self.fc_fusion(fusion_inputs)

        return predictions

# Example usage
num_users = 1000
num_items = 5000
num_numerical_features = 10
embedding_dim = 50
fusion_dim = 100
output_dim = 1

model = CfHybridBERT4Rec(num_users, num_items, num_numerical_features, fusion_dim, output_dim)

# Generate sample input data
user_ids = tf.constant([1, 2, 3, 4])  # User IDs (batch size of 4)
item_ids = tf.constant([10, 20, 30, 40])  # Item IDs
numerical_features = tf.random.normal((4, num_numerical_features))  # Numerical features (batch size x num_numerical_features)
text_inputs = tf.random.normal((4, max_seq_length))  # Textual inputs (batch size x max_seq_length)

# Forward pass
predictions = model(user_ids, item_ids, numerical_features, text_inputs)
print(predictions)
