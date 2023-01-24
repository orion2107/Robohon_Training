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
    tokenizer = transformers.BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    bert_base = transformers.TFBertModel.from_pretrained('onlplab/alephbert-base')
    # model = transformers.TFBertModel.from_pretrained('onlplab/alephbert-base')
    model = tf.keras.models.load_model('../model/alephbert_finetuned_model_v2',
                                       custom_objects={'TFBertModel': bert_base},
                                       compile=False)  # need to change model path upon deployment
    df = pd.read_csv('../data/default_sentence_list_utf8.csv') # need to change default_list.cdv path upon deployment
    default_list = df['default sentence list']

def run(raw_data):
    try:
        raw_data = json.loads(raw_data)['data']

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


        # default_list = [
        #     'מה הפעולות שאתה יודע לעשות?',
        #     'איך אתה מרגיש?',
        #     'האם עשית משהו מעניין היום?',
        #     'אני מרגיש טוב',
        #     'בגיל המבוגר חשוב במיוחד לשמור על הבריאות ולאמץ אורח חיים בריא',
        #     'מה קרה לך?',
        #     'על מה אתה מדבר?',
        #     'מה התחביבים שלך?',
        #     'מה תרצה ללמוד?',
        #     'איך פוגשים ומכירים חברים חדשים?',
        #     'מתי ביקרת אצל הרופא?',
        #     'מה קראת לאחרונה?',
        #     'תספר לי על המשפחה שלך',
        #     'במה עבדת?',
        #     'אילו בעלי חיים אתה אוהב?',
        #     'מה אתה אוהב לאכול?',
        #     'איזו מוזיקה אתה אוהב?',
        #     'איפה טיילת מחוץ לארץ ?',
        #     'מה זכרונות ילדות שלך?',
        #     'איזה חג אתה אוהב לחגוג?',
        #     'איפה אתה מבלה בטבע?',
        #     'איפה ביקרת במדינה?',
        #     'איך אתה מסתדר עם מזג האוויר?',
        #     'מה המשמעות של השם שלך?',
        #     'מה היא לדעתך הזדקנות מוצלחת?'
        # ]
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