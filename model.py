import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Load your data
data = pd.read_csv('expense-tracker\expense_classification_randomized.csv')  # Make sure 'expenses.csv' has columns 'Expense' and 'Category'
data = data.dropna()

# Encode categories
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Split into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Train Word2Vec embeddings
sentences = [row.split() for row in data['Expense']]
word2vec = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
embedding_dim = word2vec.vector_size

# Custom dataset class for PyTorch
class ExpenseDataset(Dataset):
    def __init__(self, data, word2vec, max_len=20):
        self.data = data
        self.word2vec = word2vec
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        expense_text = self.data.iloc[idx]['Expense'].split()
        category = self.data.iloc[idx]['Category']

        # Embed words and pad/truncate to max_len
        embedding_matrix = np.zeros((self.max_len, embedding_dim))
        for i, word in enumerate(expense_text[:self.max_len]):
            if word in self.word2vec.wv:
                embedding_matrix[i] = self.word2vec.wv[word]
        
        return torch.tensor(embedding_matrix, dtype=torch.float32), torch.tensor(category, dtype=torch.long)

# Create dataset and dataloaders
train_dataset = ExpenseDataset(train_data, word2vec)
val_dataset = ExpenseDataset(val_data, word2vec)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
class ExpenseClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(ExpenseClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Use hidden state from LSTM
        x = hidden[-1]  # Take the last layer's output of LSTM
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model parameters
hidden_dim = 64
num_classes = len(label_encoder.classes_)

# Initialize the model, loss function, and optimizer
model = ExpenseClassifier(embedding_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the model
torch.save(model, 'expense_classifier_model.pth')
