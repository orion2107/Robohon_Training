import numpy as np
import pandas as pd
import json
import transformers
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging
logging.set_verbosity_error()
import warnings
warnings.filterwarnings('ignore')

def init():
    global model
    global tokenizer
    global default_list
    global labels
    tokenizer = transformers.BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    bert_base = transformers.TFBertModel.from_pretrained('onlplab/alephbert-base')
    # model = transformers.TFBertModel.from_pretrained('onlplab/alephbert-base')
    model = tf.keras.models.load_model('../model/alephbert_finetuned_model_v2', custom_objects={'TFBertModel': bert_base},compile=False)
    df = pd.read_csv('path_to_csv_file') #pd.read_csv('../data/default_sentence_list_utf8.csv')
    default_list = df['default sentence list']
    labels = ["negative", "positive"]


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=32,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use onlplab/alephbert-base pretrained model.

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()
        self.max_length = 128

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)

def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )
    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    pred = labels[idx]
    return pred, proba


def reference_similarity(user_sentence, threshold, default_list):
    all_score = []
    most_similar_sentence = ""
    similarity_type = 0

    # comparing reference sentence with default list
    for idx in range(len(default_list)):
        reference_sentence = default_list[idx]
        all_score.append(check_similarity(user_sentence, reference_sentence)[1][1])

    # checking the maximum score and return sentence for maximum score even if belos threshold
    max_val = max(all_score)
    item_idx = np.where(all_score == max_val)
    item_idx = item_idx[0][0]
    most_similar_sentence = default_list[item_idx]

    if max_val >= threshold:
        similarity_type = 1  # to check if the score is above the threshold

    return (most_similar_sentence, similarity_type, max_val)


def reference_similarity_new(user_sentence, threshold, default_list):
    all_score = []
    most_similar_sentence = ""
    similarity_type = 0

    # comparing reference sentence with default list
    for idx in range(len(default_list)):
        reference_sentence = default_list[idx]
        current_score = check_similarity(user_sentence, reference_sentence)[1][1]
        all_score.append(current_score)
        if current_score >= threshold:
            similarity_type = 1
            return (reference_sentence, similarity_type, current_score)

    # checking the maximum score and return sentence for maximum score even if belos threshold
    max_val = max(all_score)
    item_idx = np.where(all_score == max_val)
    item_idx = item_idx[0][0]
    most_similar_sentence = default_list[item_idx]

    return (most_similar_sentence, similarity_type, max_val)

def run_old(raw_data):
    try:
        raw_data = json.loads(raw_data)['data']

        def check_similarity(sentence1, sentence2):
            sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
            test_data = BertSemanticDataGenerator(
                sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
            )
            proba = model.predict(test_data[0])[0]
            idx = np.argmax(proba)
            pred = labels[idx]
            return pred, proba

        ###==============================================================================================
        def reference_similarity(user_sentence,threshold,default_list):
            all_score = []
            most_similar_sentence = ""
            similarity_type = 0

            # initialize dictionary that will contain tokenized sentences
            tokens = {'input_ids': [], 'attention_mask': []}


            # comparing reference sentence with default list
            for idx in range(len(default_list)+1):
                if idx == 0:
                    new_tokens = tokenizer.encode_plus(user_sentence, max_length=128, truncation=True,
                                        padding='max_length', return_tensors='pt')
                    tokens['input_ids'].append(new_tokens['input_ids'][0])
                    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
                else:
                    new_tokens = tokenizer.encode_plus(default_list[idx -1], max_length=128, truncation=True,
                                        padding='max_length', return_tensors='pt')
                    tokens['input_ids'].append(new_tokens['input_ids'][0])
                    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

            # reformat list of tensors into single tensor
            tokens['input_ids'] = tf.stack(tokens['input_ids'])
            tokens['attention_mask'] = tf.stack(tokens['attention_mask'])

            outputs = model(tokens)
            embeddings = outputs.last_hidden_state
            attention_mask = tokens['attention_mask']
            mask = tf.expand_dims(attention_mask,-1)
            mask = tf.broadcast_to(mask, embeddings.shape)
            mask = np.asarray(mask).astype(np.float32)
            masked_embeddings = embeddings * mask
            summed = tf.math.reduce_sum(masked_embeddings, 1)
            summed_mask = tf.clip_by_value(mask.sum(1), clip_value_min=1e-9, clip_value_max=1000000)
            mean_pooled = summed / summed_mask
            # convert from PyTorch tensor to numpy array
            mean_pooled = mean_pooled.numpy()

            # calculate similarity by comparing 0th value to the rest
            all_score = cosine_similarity(
                [mean_pooled[0]],
                mean_pooled[1:]
            )
            all_score = all_score[0]  # from 2D list to 1D

            #find the max value (similarity) and return the most similar sentence
            max_val = max(all_score)
            item_idx = np.where(all_score==max_val)
            item_idx = item_idx[0][0]
            most_similar_sentence = default_list[item_idx]

        
            if max_val >= threshold:
                similarity_type = 1   # to check if the score is above the threshold


            return (most_similar_sentence, similarity_type, max_val)

        #--------------------------------------------------------------------------------------------------------

        most_similar_sentence, similarity_type, max_val = reference_similarity(raw_data[0],raw_data[1],default_list)

        all_vals = []
        all_vals.append(most_similar_sentence)
        all_vals.append(similarity_type)
        all_vals.append(float(max_val))
    except:
        all_vals = [0,0,0]

    return all_vals
  ###----------------------------------------------------------------------------------------------

def run(raw_data):
    try:
        raw_data = json.loads(raw_data)['data']

        def check_similarity(sentence1, sentence2):
            sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
            test_data = BertSemanticDataGenerator(
                sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
            )
            proba = model.predict(test_data[0])[0]
            idx = np.argmax(proba)
            pred = labels[idx]
            return pred, proba

        ###==============================================================================================
        def reference_similarity(user_sentence, threshold, default_list):
            all_score = []
            most_similar_sentence = ""
            similarity_type = 0

            # initialize dictionary that will contain tokenized sentences
            tokens = {'input_ids': [], 'attention_mask': []}

            # comparing reference sentence with default list
            for idx in range(len(default_list) + 1):
                if idx == 0:
                    new_tokens = tokenizer.encode_plus(user_sentence, max_length=128, truncation=True,
                                                       padding='max_length', return_tensors='pt')
                    tokens['input_ids'].append(new_tokens['input_ids'][0])
                    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
                else:
                    new_tokens = tokenizer.encode_plus(default_list[idx - 1], max_length=128, truncation=True,
                                                       padding='max_length', return_tensors='pt')
                    tokens['input_ids'].append(new_tokens['input_ids'][0])
                    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

            # reformat list of tensors into single tensor
            tokens['input_ids'] = tf.stack(tokens['input_ids'])
            tokens['attention_mask'] = tf.stack(tokens['attention_mask'])

            outputs = model(tokens)
            embeddings = outputs.last_hidden_state
            attention_mask = tokens['attention_mask']
            mask = tf.expand_dims(attention_mask, -1)
            mask = tf.broadcast_to(mask, embeddings.shape)
            mask = np.asarray(mask).astype(np.float32)
            masked_embeddings = embeddings * mask
            summed = tf.math.reduce_sum(masked_embeddings, 1)
            summed_mask = tf.clip_by_value(mask.sum(1), clip_value_min=1e-9, clip_value_max=1000000)
            mean_pooled = summed / summed_mask
            # convert from PyTorch tensor to numpy array
            mean_pooled = mean_pooled.numpy()

            # calculate similarity by comparing 0th value to the rest
            all_score = cosine_similarity(
                [mean_pooled[0]],
                mean_pooled[1:]
            )
            all_score = all_score[0]  # from 2D list to 1D

            # find the max value (similarity) and return the most similar sentence
            max_val = max(all_score)
            item_idx = np.where(all_score == max_val)
            item_idx = item_idx[0][0]
            most_similar_sentence = default_list[item_idx]

            if max_val >= threshold:
                similarity_type = 1  # to check if the score is above the threshold

            return (most_similar_sentence, similarity_type, max_val)
        # --------------------------------------------------------------------------------------------------------

        most_similar_sentence, similarity_type, max_val = reference_similarity(raw_data[0], raw_data[1], default_list)

        all_vals = []
        all_vals.append(most_similar_sentence)
        all_vals.append(similarity_type)
        all_vals.append(float(max_val))
    except:
        all_vals = [0, 0, 0]

    return all_vals
###----------------------------------------------------------------------------------------------