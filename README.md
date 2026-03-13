# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset

The goal is to build a Named Entity Recognition (NER) system.
It should identify entities like persons, organizations, locations, and dates in text.
The model will be trained on a tagged dataset of words and their corresponding entity labels.
Input sentences need to be encoded and padded for model processing.
A BiLSTM neural network will be used to learn sequential patterns in the text.
The system should predict entity labels for unseen sentences and evaluate its accuracy.


<img width="410" height="491" alt="image-3" src="https://github.com/user-attachments/assets/5a14de82-ce0a-4d03-9c0f-ae948afe721d" />

## DESIGN STEPS

### STEP 1:

Load the NER dataset, fill missing values, and identify unique words and tags.

### STEP 2:

Create mappings for words and tags to numeric indices, and reverse mappings for decoding.

### STEP 3:

Group words and tags by sentences using a custom `SentenceGetter` class.

### STEP 4:

Encode words and tags as sequences, pad them to a fixed length, and split into training and test sets.

### STEP 5:

Define a PyTorch Dataset and DataLoader, build the BiLSTM model, and set up loss and optimizer.

### STEP 6:

Train the model, evaluate on test data, and visualize training and validation loss over epochs.


## PROGRAM
### Name: Ahil Santo A
### Register Number: 212224040018
```python
# Function for Model
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size,tagset_size, embedding_dim=50, hidden_dim=100):
    super(BiLSTMTagger,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,tagset_size)

  def forward(self, input_ids):
      x = self.embedding(input_ids)
      x = self.dropout(x)
      x, _ = self.lstm(x)
      return self.fc(x)

# Model, Loss function, Optimizer
model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        model.eval()
        val_loss=0
        with torch.no_grad():
            for batch in test_loader:
              input_ids = batch["input_ids"].to(device)
              labels = batch["labels"].to(device)
              outputs = model(input_ids)
              loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
              val_loss += loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f} Val Loss = {val_loss:.4f} ")
    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="578" height="71" alt="image-2" src="https://github.com/user-attachments/assets/f967fae2-d484-48c5-8e02-1eb6c7eb3830" />

<img width="847" height="672" alt="image" src="https://github.com/user-attachments/assets/43fe084e-6639-40c6-bab5-e2120e6e651b" />

### Sample Text Prediction

<img width="561" height="492" alt="image-1" src="https://github.com/user-attachments/assets/75dde5ed-c3b1-49a9-b34e-88c74f315b7f" />


## RESULT


Thus the program to develop an LSTM-based model for recognizing the named entities in the text is developed successfully.
