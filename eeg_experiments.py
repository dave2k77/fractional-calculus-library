#!/usr/bin/env python3
"""
EEG Classification Experiments for hpfracc
Phase 1: Real EEG Data Collection and Analysis

Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)
Dataset: BCI Competition IV Dataset 2a (when available) + MNE sample data
Goal: Replace synthetic 91.5% vs 87.6% with real experimental results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys
from pathlib import Path

# Add hpfracc to path
sys.path.append('/home/davianc/fractional-calculus-library')
from hpfracc.ml import FractionalNeuralNetwork, FractionalConv1D, LayerConfig
from hpfracc.core.definitions import FractionalOrder

class EEGDataLoader:
    """Load and preprocess EEG data for classification experiments"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.raw_data = None
        self.epochs = None
        self.X = None
        self.y = None
        self.dataset_info = None
        
        # Load dataset documentation
        self.load_dataset_documentation()
    
    def load_dataset_documentation(self):
        """Load dataset documentation from JSON files"""
        try:
            import json
            doc_path = Path('dataset_documentation/all_datasets_documentation.json')
            if doc_path.exists():
                with open(doc_path, 'r') as f:
                    self.dataset_info = json.load(f)
                print("Loaded dataset documentation")
            else:
                print("Dataset documentation not found - run dataset_documentation.py first")
                self.dataset_info = {}
        except Exception as e:
            print(f"Error loading dataset documentation: {e}")
            self.dataset_info = {}
        
    def load_sample_data(self):
        """Load MNE sample data for initial testing"""
        print("Loading MNE sample EEG data...")
        
        # Load sample data from MNE
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample', 
                                          'sample_audvis_raw.fif')
        
        # Load raw data
        self.raw_data = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
        
        # Create events for classification (auditory vs visual)
        events = mne.find_events(self.raw_data, stim_channel='STI 014')
        
        # Create epochs
        event_id = {'auditory/left': 1, 'auditory/right': 2, 
                   'visual/left': 3, 'visual/right': 4}
        tmin, tmax = -0.2, 0.5
        
        self.epochs = mne.Epochs(self.raw_data, events, event_id, tmin, tmax,
                               baseline=(None, 0), preload=True)
        
        print(f"Loaded {len(self.epochs)} epochs")
        print(f"Channels: {len(self.epochs.ch_names)}")
        print(f"Time points: {len(self.epochs.times)}")
        
        return self.epochs
    
    def load_bci_data(self, data_path):
        """Load BCI Competition IV Dataset 2a when available"""
        print(f"Loading BCI data from {data_path}...")
        
        # This will be implemented when we get the actual BCI dataset
        # For now, we'll use sample data
        return self.load_sample_data()
    
    def preprocess_data(self, epochs):
        """Preprocess EEG data for classification"""
        print("Preprocessing EEG data...")
        
        # Apply bandpass filter
        epochs.filter(8, 30, fir_design='firwin')
        
        # Get data as numpy array
        data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        # Reshape for classification
        n_epochs, n_channels, n_times = data.shape
        X = data.reshape(n_epochs, -1)  # Flatten to (n_epochs, n_channels * n_times)
        
        # Get labels
        y = epochs.events[:, 2]  # Event IDs
        
        # Encode labels to start from 0
        unique_labels = np.unique(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_encoded = np.array([label_mapping[label] for label in y])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.X = X_scaled
        self.y = y_encoded
        
        print(f"Preprocessed data shape: {X_scaled.shape}")
        print(f"Original labels: {np.unique(y)}")
        print(f"Encoded labels: {np.unique(y_encoded)}")
        
        return X_scaled, y_encoded

class StandardCNN(nn.Module):
    """Standard CNN for EEG classification"""
    
    def __init__(self, input_channels, input_length, num_classes):
        super(StandardCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_channels, input_length)
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def _get_flattened_size(self, input_channels, input_length):
        """Calculate the size after conv layers"""
        x = torch.zeros(1, input_channels, input_length)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.numel()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class FractionalEEGNet(nn.Module):
    """Fractional Neural Network for EEG classification using hpfracc"""
    
    def __init__(self, input_channels, input_length, num_classes, alpha=0.5):
        super(FractionalEEGNet, self).__init__()
        
        self.alpha = alpha
        
        # Create layer config with fractional order
        config = LayerConfig(fractional_order=FractionalOrder(alpha))
        
        # Fractional convolution layers
        self.frac_conv1 = FractionalConv1D(input_channels, 32, kernel_size=3, 
                                          padding=1, config=config)
        self.frac_conv2 = FractionalConv1D(32, 64, kernel_size=3, 
                                          padding=1, config=config)
        self.frac_conv3 = FractionalConv1D(64, 128, kernel_size=3, 
                                          padding=1, config=config)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.flattened_size = self._get_flattened_size(input_channels, input_length)
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def _get_flattened_size(self, input_channels, input_length):
        """Calculate the size after conv layers"""
        x = torch.zeros(1, input_channels, input_length)
        x = self.pool(torch.relu(self.frac_conv1(x)))
        x = self.pool(torch.relu(self.frac_conv2(x)))
        x = self.pool(torch.relu(self.frac_conv3(x)))
        return x.numel()
    
    def forward(self, x):
        x = self.relu(self.frac_conv1(x))
        x = self.pool(x)
        x = self.relu(self.frac_conv2(x))
        x = self.pool(x)
        x = self.relu(self.frac_conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class EEGExperimentRunner:
    """Run EEG classification experiments and collect real results"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Hardware info
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    def run_experiments(self, X, y, epochs_data):
        """Run comprehensive EEG classification experiments"""
        print("\n" + "="*60)
        print("EEG CLASSIFICATION EXPERIMENTS")
        print("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)")
        print("="*60)
        
        results = {}
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get data dimensions
        n_channels = epochs_data.info['nchan']
        n_times = len(epochs_data.times)
        num_classes = len(np.unique(y))
        
        print(f"Data dimensions: {n_channels} channels, {n_times} time points")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Number of classes: {num_classes}")
        
        # 1. Standard Machine Learning Methods
        print("\n1. STANDARD MACHINE LEARNING METHODS")
        print("-" * 40)
        
        # Random Forest
        start_time = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_time = time.time() - start_time
        
        print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_time:.2f}s)")
        results['Random Forest'] = {'accuracy': rf_accuracy, 'time': rf_time}
        
        # SVM
        start_time = time.time()
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        svm_time = time.time() - start_time
        
        print(f"SVM Accuracy: {svm_accuracy:.4f} ({svm_time:.2f}s)")
        results['SVM'] = {'accuracy': svm_accuracy, 'time': svm_time}
        
        # 2. Standard CNN
        print("\n2. STANDARD CNN")
        print("-" * 40)
        
        # Prepare data for CNN
        X_train_cnn = X_train.reshape(-1, n_channels, n_times)
        X_test_cnn = X_test.reshape(-1, n_channels, n_times)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_cnn).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_cnn).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train Standard CNN
        cnn_model = StandardCNN(n_channels, n_times, num_classes).to(self.device)
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        cnn_criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        cnn_model.train()
        for epoch in range(50):  # Reduced epochs for faster testing
            for batch_X, batch_y in train_loader:
                cnn_optimizer.zero_grad()
                outputs = cnn_model(batch_X)
                loss = cnn_criterion(outputs, batch_y)
                loss.backward()
                cnn_optimizer.step()
        
        # Test Standard CNN
        cnn_model.eval()
        cnn_correct = 0
        cnn_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = cnn_model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                cnn_total += batch_y.size(0)
                cnn_correct += (predicted == batch_y).sum().item()
        
        cnn_accuracy = cnn_correct / cnn_total
        cnn_time = time.time() - start_time
        
        print(f"Standard CNN Accuracy: {cnn_accuracy:.4f} ({cnn_time:.2f}s)")
        results['Standard CNN'] = {'accuracy': cnn_accuracy, 'time': cnn_time}
        
        # 3. Fractional Neural Network
        print("\n3. FRACTIONAL NEURAL NETWORK")
        print("-" * 40)
        
        # Train Fractional Neural Network
        frac_model = FractionalEEGNet(n_channels, n_times, num_classes, alpha=0.5)
        
        # Move all components to device
        frac_model = frac_model.to(self.device)
        for name, module in frac_model.named_modules():
            if hasattr(module, 'to'):
                module.to(self.device)
        
        frac_optimizer = optim.Adam(frac_model.parameters(), lr=0.001)
        frac_criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        frac_model.train()
        for epoch in range(50):  # Reduced epochs for faster testing
            for batch_X, batch_y in train_loader:
                frac_optimizer.zero_grad()
                outputs = frac_model(batch_X)
                loss = frac_criterion(outputs, batch_y)
                loss.backward()
                frac_optimizer.step()
        
        # Test Fractional Neural Network
        frac_model.eval()
        frac_correct = 0
        frac_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = frac_model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                frac_total += batch_y.size(0)
                frac_correct += (predicted == batch_y).sum().item()
        
        frac_accuracy = frac_correct / frac_total
        frac_time = time.time() - start_time
        
        print(f"Fractional Neural Network Accuracy: {frac_accuracy:.4f} ({frac_time:.2f}s)")
        results['Fractional Neural Network'] = {'accuracy': frac_accuracy, 'time': frac_time}
        
        # 4. Results Summary
        print("\n4. RESULTS SUMMARY")
        print("-" * 40)
        print("Method                    | Accuracy | Time (s)")
        print("-" * 40)
        for method, result in results.items():
            print(f"{method:<25} | {result['accuracy']:.4f}   | {result['time']:.2f}")
        
        # 5. Statistical Analysis
        print("\n5. STATISTICAL ANALYSIS")
        print("-" * 40)
        
        # Compare Standard CNN vs Fractional Neural Network
        cnn_acc = results['Standard CNN']['accuracy']
        frac_acc = results['Fractional Neural Network']['accuracy']
        
        print(f"Standard CNN Accuracy: {cnn_acc:.4f}")
        print(f"Fractional Neural Network Accuracy: {frac_acc:.4f}")
        print(f"Improvement: {frac_acc - cnn_acc:.4f} ({((frac_acc - cnn_acc) / cnn_acc * 100):.2f}%)")
        
        # Save results
        self.save_results(results, epochs_data)
        
        return results
    
    def save_results(self, results, epochs_data):
        """Save experimental results"""
        print("\n6. SAVING RESULTS")
        print("-" * 40)
        
        # Create results directory
        os.makedirs('eeg_results', exist_ok=True)
        
        # Save results to file
        with open('eeg_results/real_eeg_results.txt', 'w') as f:
            f.write("EEG Classification Results\n")
            f.write("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)\n")
            f.write("Dataset: MNE Sample Data (BCI Competition IV when available)\n")
            f.write("=" * 50 + "\n\n")
            
            for method, result in results.items():
                f.write(f"{method}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Time: {result['time']:.2f}s\n\n")
        
        # Create visualization
        self.create_visualization(results)
        
        print("Results saved to eeg_results/real_eeg_results.txt")
        print("Visualization saved to eeg_results/eeg_results_plot.png")
    
    def create_visualization(self, results):
        """Create visualization of results"""
        methods = list(results.keys())
        accuracies = [results[method]['accuracy'] for method in methods]
        times = [results[method]['time'] for method in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        bars1 = ax1.bar(methods, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_title('EEG Classification Accuracy Comparison\nASUS TUF A15 (RTX 3050, 30GB RAM)', fontsize=14)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom')
        
        # Time plot
        bars2 = ax2.bar(methods, times, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax2.set_title('Training Time Comparison\nASUS TUF A15 (RTX 3050, 30GB RAM)', fontsize=14)
        ax2.set_ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('eeg_results/eeg_results_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run EEG experiments"""
    print("Starting EEG Classification Experiments...")
    print("Phase 1: Real Data Collection on ASUS TUF A15")
    print("Goal: Replace synthetic 91.5% vs 87.6% with real results")
    
    # Initialize data loader
    data_loader = EEGDataLoader()
    
    # Load sample data (will be replaced with BCI Competition IV when available)
    epochs = data_loader.load_sample_data()
    
    # Preprocess data
    X, y = data_loader.preprocess_data(epochs)
    
    # Initialize experiment runner
    runner = EEGExperimentRunner()
    
    # Run experiments
    results = runner.run_experiments(X, y, epochs)
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("Real EEG classification results collected.")
    print("Ready to update manuscript with honest data.")
    print("="*60)

if __name__ == "__main__":
    main()
