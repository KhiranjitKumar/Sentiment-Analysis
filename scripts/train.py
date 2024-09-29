import torch
from torch.utils.data import DataLoader, TensorDataset
from model import BERTBiLSTMClassifier
from preprocess import split_and_tokenize_data, load_data
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loaders(train_data, val_data=None):
    train_input_ids = train_data[0]['input_ids']
    train_attention_masks = train_data[0]['attention_mask']
    train_labels = torch.tensor(train_data[1])

    print(f"Train input_ids shape: {train_input_ids.shape}")
    print(f"Train attention_masks shape: {train_attention_masks.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataloader = None
    if val_data:
        val_input_ids = val_data[0]['input_ids']
        val_attention_masks = val_data[0]['attention_mask']
        val_labels = torch.tensor(val_data[1])

        print(f"Validation input_ids shape: {val_input_ids.shape}")
        print(f"Validation attention_masks shape: {val_attention_masks.shape}")
        print(f"Validation labels shape: {val_labels.shape}")

        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=32)

    return train_dataloader, val_dataloader

def evaluate(model, val_dataloader):
    model.eval()
    total_val_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_dataloader)

def train_model():
    data_dir = r'K:\sentiment-classification\data'
    pos_reviews, neg_reviews = load_data(data_dir)
    train_data, val_data, _ = split_and_tokenize_data(pos_reviews, neg_reviews)

    print("Data loading and tokenization completed.")
    print(f"Number of training samples: {len(train_data[0]['input_ids'])}")
    print(f"Number of validation samples: {len(val_data[0]['input_ids'])}")

    train_dataloader, val_dataloader = create_data_loaders(train_data, val_data)

    model = BERTBiLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_path = r'K:\sentiment-classification\logs\best_model.pth'

    for epoch in range(3):
        model.train()
        total_train_loss = 0
        print(f"Starting epoch {epoch+1}")
        
        for batch in train_dataloader:
            input_ids, attention_masks, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            print(f"Batch Loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss}")

        if val_dataloader:
            val_loss = evaluate(model, val_dataloader)
            print(f"Epoch {epoch+1} Validation Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch+1} with validation loss: {val_loss}")

    print(f"Training completed. Best model saved with validation loss: {best_val_loss}")

if __name__ == '__main__':
    train_model()
