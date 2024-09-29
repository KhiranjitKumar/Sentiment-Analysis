import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from model import BERTBiLSTMClassifier
from preprocess import split_and_tokenize_data, load_data
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_on_test():
    model = BERTBiLSTMClassifier().to(device)
    model.load_state_dict(torch.load(r'K:\sentiment-classification\logs\best_model.pth'))
    model.eval()
    
    data_dir = r'K:\sentiment-classification\data'
    pos_reviews, neg_reviews = load_data(data_dir)
    _, _, test_data = split_and_tokenize_data(pos_reviews, neg_reviews)
    
    test_input_ids = test_data[0]['input_ids']
    test_attention_masks = test_data[0]['attention_mask']
    test_labels = torch.tensor(test_data[1])
    
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_masks, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_masks)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    evaluate_on_test()
