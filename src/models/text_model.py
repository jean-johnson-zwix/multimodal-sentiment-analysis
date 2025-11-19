import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer


class TextualSentimentModel(nn.Module):
    """
    Implementation of RoBERTA-LSTM model for texual sentiment analysis.
    """

    def __init__(self,
                 num_classes=3,
                 input_size=768):
        """
        Args:
    `       TODO
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.roberta = AutoModel.from_pretrained('roberta-base')

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True)

        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(256, 80),
                                        nn.Linear(80, 20),
                                        nn.Linear(20, self.num_classes),
                                        nn.Softmax(dim=1))

        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        """
        TODO docstring
        """
        raw_outputs = self.roberta(
            inputs['input_ids'], inputs['attention_mask'])
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.classifier(outputs)
        return outputs


def calculate_ground_truth(row):
    """
    Function to calculate ground truth labels. This does not account for rows with conflicting labels. 
    """
    text_label = row['text_label']
    image_label = row['image_label']
    if text_label == image_label:
        ground_truth_label = text_label
    elif text_label == 'neutral':
        ground_truth_label = image_label
    elif image_label == 'neutral':
        ground_truth_label = text_label
    else:
        ground_truth_label = 'neutral'
    return ground_truth_label
        
class TextualDataset(Dataset):
    """
    Text-only dataset with text-label
    """

    def __init__(self, base_dir, tokenizer):
        """
        Args:
            base_dir (string): Path to base directory containing data
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

        self.raw_text = self.load_text()

    def load_text(self) -> pd.DataFrame:
        """
        Loads text-only data as pandas dataframe with id, text, label
        """
        labels_file = os.path.join(self.base_dir, "labelResultAll.txt")

        # Load the file (tab-separated)
        df = pd.read_csv(labels_file, sep="\t")

        # Split the 'text,image' column into two separate columns
        df[['text_label', 'image_label']] = df['text,image'].str.split(',', expand=True)

        # Optionally drop the original combined column
        df = df.drop(columns=['text,image'])

        # Remove conflicting labels
        condition = ((df['text_label'] == 'positive') & (df['image_label'] == 'negative')) | ((df['text_label'] == 'negative') & (df['image_label'] == 'positive'))
        df = df.drop(df[condition].index)

        df['ground_truth'] = df.apply(calculate_ground_truth, axis=1)    
        
        return df

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.raw_text.iloc[idx]
        item_id = row['ID']
        label_str = row['ground_truth']
        text_path = os.path.join(self.data_dir, f"{item_id}.txt")

        with open(text_path, "r", encoding="latin-1") as f:
            text = f.read().strip()

        encoding = self.tokenizer(text, return_tensors='pt')

        label = torch.tensor(self.label_map[label_str])

        encoding['input_ids'] = encoding['input_ids'].squeeze(0)
        encoding['attention_mask'] = encoding['attention_mask'].squeeze(0)

        return encoding, label
    
def collate(batch):
    """
    Ensures that the batch is formatted correctly including adding necessary padding.
    """
    inputs, labels = list(zip(*batch))

    labels = torch.stack(labels)
    labels = torch.nn.functional.one_hot(labels, num_classes=3).float()

    # Extract input_ids and attention_masks
    input_ids = [x['input_ids'] for x in inputs]
    attention_masks = [x['attention_mask'] for x in inputs]

    # Pad them to the same length
    padded_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0)

    # Combine into one dict
    batch_padded = {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks
    }

    return batch_padded, labels


def test_textual_model():
    batch_size = 32

    text_dataset = TextualDataset('notebooks/data/raw/mvsa-single/MVSA_Single')
    dataset_size = len(text_dataset)
    train_size = int(0.8 * dataset_size)
    validation_size = dataset_size - train_size
    train_data, val_data = random_split(
        text_dataset, [train_size, validation_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, collate_fn=collate)

    num_epochs = 30
    learning_rate = 1e-5
    device='cuda' if torch.cuda.is_available() else 'cpu'

    model = TextualSentimentModel()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting TRAINING for epoch {epoch+1}...")
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Send inputs and labels to device
            for key, value in inputs.items():
                inputs[key] = inputs[key].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # Validation loop
        print(f"Starting VALIDATION for epoch {epoch+1}...")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Send inputs and labels to device
                for key, value in inputs.items():
                    inputs[key] = inputs[key].to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, correct_label = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == correct_label).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val Acc: {accuracy:.2f}%")

    torch.save(model.state_dict(), "text_model.pt")

if __name__ == "__main__":
    test_textual_model()