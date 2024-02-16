"""--------------------------------------------------"""
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        
        Parameters:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        """
        Call this function after each epoch to check if the training should be stopped.

        Parameters:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if not self.early_stop:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            torch.save(model.state_dict(), 'best_dl_model.pt')
            self.val_loss_min = val_loss





"""--------------------------------------------------"""
import time
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

"""Train the model for one epoch"""
def train_epoch(model, train_loader, criterion, optimizer):
    """
    Trains the model for one epoch using the given data loader, criterion, and optimizer.
    
    Args:
    model: The PyTorch model to train.
    train_loader: The PyTorch data loader for the training data.
    criterion: The loss function to use.
    optimizer: The optimizer to use to update the model parameters.
    
    Returns:
    The average training loss for the epoch.
    """
    
    # Set the model to training mode
    model.train()
    
    # Initialize the total training loss for the epoch
    total_train_loss = 0.0
    
    # Iterate over the training data loader using a progress bar
    for data in tqdm(train_loader, desc='Training', leave=False):
        # Unpack the data into statements and labels
        statement, labels = data
        
        # Clear the gradients accumulated in the previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute the model outputs for the current batch
        outputs = model(statement)
        
        # Calculate the loss between the model outputs and the ground truth labels
        loss = criterion(outputs, labels)
        
        # Backward pass: compute the gradients of the loss with respect to the model parameters
        loss.backward()
        
        # Update the model parameters using the optimizer
        optimizer.step()
        
        # Accumulate the loss for the current batch
        total_train_loss += loss.item()
    
    # Return the average training loss for the epoch
    return total_train_loss / len(train_loader)

"""Validate the model for one epoch"""
def validate_epoch(model, val_loader, criterion):
    """
    Evaluates the model on the validation set using the given data loader and criterion.
    
    Args:
    model: The PyTorch model to evaluate.
    val_loader: The PyTorch data loader for the validation data.
    criterion: The loss function to use.
    
    Returns:
    A tuple containing:
      - The average validation loss for the epoch.
      - A list of all predicted labels by the model.
      - A list of all true labels in the validation set.
    """
    
    # Put the model in evaluation mode
    model.eval()
    
    # Initialize the total validation loss for the epoch
    total_val_loss = 0.0
    
    # Initialize empty lists to store predicted and true labels
    all_predicted_labels = []
    all_true_labels = []
    
    # Disable gradient calculation for efficiency during evaluation
    with torch.no_grad():
        
        # Iterate over the validation data loader using a progress bar
        for data in tqdm(val_loader, desc='Validation', leave=False):
            # Unpack the data into statements and labels
            statement, labels = data
            
            # Forward pass: compute the model outputs for the current batch
            outputs = model(statement)
            
            # Calculate the loss between the model outputs and the ground truth labels
            loss = criterion(outputs, labels)
            
            # Accumulate the loss for the current batch
            total_val_loss += loss.item()
            
            # Convert predicted labels to numpy array and store them
            predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predicted_labels.extend(predicted_labels)
            
            # Convert true labels to numpy array and store them
            true_labels = labels.cpu().numpy()
            all_true_labels.extend(true_labels)
    
    # Return the average validation loss, predicted labels, and true labels
    return total_val_loss / len(val_loader), all_predicted_labels, all_true_labels


"""Calculates accuracy given predicted and true labels"""
def calculate_accuracy(predicted_labels, true_labels):
    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))

    total_samples = len(predicted_labels)

    accuracy = correct_predictions / total_samples

    return accuracy

"""Calculate accuracy, precision, recall, and F1 score"""
def calculate_metrics(true_labels, predicted_labels):
    accuracy = calculate_accuracy(predicted_labels, true_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

"""Plots the graph of accuracy, precision, recall, and F1 score against epochs"""
import matplotlib.pyplot as plt

def plot_metrics(metrics_history):
    epochs = range(1, len(metrics_history['train_loss']) + 1)

    # Define plot size
    plt.figure(figsize=(15, 10))

    # Plotting Training Loss
    plt.subplot(3, 2, 1)
    plt.plot(epochs, metrics_history['train_loss'], label='Training Loss', color='orange')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Validation Loss
    plt.subplot(3, 2, 2)
    plt.plot(epochs, metrics_history['val_loss'], label='Validation Loss', color='cyan')
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(3, 2, 3)
    plt.plot(epochs, metrics_history['accuracy'], label='Accuracy', color='blue')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Precision
    plt.subplot(3, 2, 4)
    plt.plot(epochs, metrics_history['precision'], label='Precision', color='red')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Plotting Recall
    plt.subplot(3, 2, 5)
    plt.plot(epochs, metrics_history['recall'], label='Recall', color='green')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # Plotting F1 Score
    plt.subplot(3, 2, 6)
    plt.plot(epochs, metrics_history['f1'], label='F1 Score', color='purple')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

"""Train and evaluate the model over multiple epochs"""
def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, num_epochs, plot):
    metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'train_loss': [], 'val_loss': []}

    early_stopping = EarlyStopping(patience=2, min_delta=0.001)
    
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # Evaluate
        avg_val_loss, predicted_labels, true_labels = validate_epoch(model, valid_loader, criterion)

        # Calculate metrics
        accuracy, precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)

        # Record metrics
        metrics_history['accuracy'].append(accuracy)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1'].append(f1)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)   

        # Print metrics
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, "
              f"Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}")
        
        # Call early stopping function
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Plot metrics if enabled
    if plot:
        plot_metrics(metrics_history)

    return predicted_labels, true_labels, accuracy




"""--------------------------------------------------"""
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def print_evaluation_metrics(true_labels, predicted_labels):
    """
    Prints the classification report, accuracy, precision, recall, and F1 score.

    Parameters:
    true_labels (list or array): True labels of the dataset.
    predicted_labels (list or array): Predicted labels by the model.
    """
    # Calculate and print the classification report
    class_report = classification_report(true_labels, predicted_labels)
    print("Classification Report:\n", class_report)

    # Calculate and print accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1-Score: {f1:.2f}")