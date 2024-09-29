from transformers import BertTokenizer
import os
import random

def load_data(data_dir):
    pos_file = os.path.join(data_dir, 'rt-polarity.pos')
    neg_file = os.path.join(data_dir, 'rt-polarity.neg')

    with open(pos_file, 'r', encoding='latin-1') as f:
        pos_reviews = f.readlines()
    
    with open(neg_file, 'r', encoding='latin-1') as f:
        neg_reviews = f.readlines()

    return pos_reviews, neg_reviews

def tokenize_sentences(sentences, max_len=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
    return tokenizer(sentences, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

def split_and_tokenize_data(pos, neg, shuffle=True):
    if shuffle:
        random.shuffle(pos)
        random.shuffle(neg)

    train_pos = pos[:4000]
    train_neg = neg[:4000]

    val_pos = pos[4000:4500]
    val_neg = neg[4000:4500]

    test_pos = pos[4500:]
    test_neg = neg[4500:]

   
    train_data = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)

    val_data = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)

    test_data = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    tokenized_train = tokenize_sentences(train_data)
    tokenized_val = tokenize_sentences(val_data)
    tokenized_test = tokenize_sentences(test_data)

    print(f"Tokenized training data size: {tokenized_train['input_ids'].shape}")
    print(f"Tokenized validation data size: {tokenized_val['input_ids'].shape}")
    print(f"Tokenized test data size: {tokenized_test['input_ids'].shape}")

    return (tokenized_train, train_labels), (tokenized_val, val_labels), (tokenized_test, test_labels)

if __name__ == '__main__':
    data_dir = r'K:\sentiment-classification\data'

    pos_reviews, neg_reviews = load_data(data_dir)
    train_data, val_data, test_data = split_and_tokenize_data(pos_reviews, neg_reviews)
