import torch
import torch.nn as nn
import torch.optim as optim


# Example data
X = torch.tensor([
  [1, 2, 3, 4, 5],
  [5, 4, 3, 2, 1],
  [1, 1, 1, 1, 1],
  [5, 5, 5, 5, 5]
])
# Example target classes
y = torch.tensor([0, 1, 0, 1])

class BasicRNN(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, num_classes)

  def forward(self, x):
    embedded = self.embedding(x)               # (batch, seq, embed_dim)
    out, h_n = self.rnn(embedded)              # out: all outputs, h_n: final hidden
    last_hidden = h_n.squeeze(0)               # (batch, hidden_dim)
    logits = self.fc(last_hidden)              # (batch, num_classes)
    return logits
  
# vocab_size=6
# embed_dim=10
# hidden_dim=20
# num_classes=2
model=BasicRNN(6, 10, 20, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
  optimizer.zero_grad()
  logits = model(X)
  loss = criterion(logits, y)
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 20 == 0:
    print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

with torch.no_grad():
  test_seq = torch.tensor([[1, 2, 3, 4, 5]])
  prediction = model(test_seq)
  predicted_class = torch.argmax(prediction, dim=1)
  print("Predicted class:", predicted_class.item())