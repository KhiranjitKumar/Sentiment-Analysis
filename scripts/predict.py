import torch
from model import BERTBiLSTMClassifier
from preprocess import tokenize_sentences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = BERTBiLSTMClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict(model, sentences):
    tokenized_inputs = tokenize_sentences(sentences)
    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_masks = tokenized_inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    return predictions

if __name__ == '__main__':
    model = load_model('./logs/best_model.pth')
    sample_sentences = ["the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal . "]
    preds = predict(model, sample_sentences)
    print(f"Predictions: {preds}")
