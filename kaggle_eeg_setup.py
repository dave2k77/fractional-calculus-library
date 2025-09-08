#!/usr/bin/env python3
"""
Kaggle EEG Dataset Setup for hpfracc
Phase 1: Real EEG Data Collection

This script helps set up Kaggle API and download EEG datasets for realistic experiments.
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("Setting up Kaggle API...")
    
    # Create kaggle directory
    kaggle_dir = Path.home() / '.config' / 'kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Kaggle config directory: {kaggle_dir}")
    print("\nTo use Kaggle API, you need to:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print(f"4. Place it in: {kaggle_dir / 'kaggle.json'}")
    print("\nAlternatively, you can set environment variables:")
    print("export KAGGLE_USERNAME=your_username")
    print("export KAGGLE_KEY=your_api_key")
    
    return kaggle_dir

def search_eeg_datasets():
    """Search for EEG datasets on Kaggle"""
    print("\nSearching for EEG datasets on Kaggle...")
    
    # Common EEG dataset searches
    searches = [
        "BCI Competition IV",
        "EEG motor imagery",
        "brain computer interface",
        "EEG classification",
        "motor imagery EEG"
    ]
    
    print("Available EEG datasets on Kaggle:")
    print("-" * 50)
    
    for search_term in searches:
        print(f"\nSearching for: {search_term}")
        try:
            # This would work once Kaggle API is set up
            result = subprocess.run([
                'kaggle', 'datasets', 'list', '-s', search_term, '--csv'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # More than just header
                    print(f"Found {len(lines)-1} datasets:")
                    for line in lines[1:3]:  # Show first 2 results
                        parts = line.split(',')
                        if len(parts) >= 2:
                            print(f"  - {parts[1]} (by {parts[2]})")
                else:
                    print("  No datasets found")
            else:
                print(f"  Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("  Search timed out")
        except Exception as e:
            print(f"  Error: {e}")

def download_eeg_dataset(dataset_name):
    """Download a specific EEG dataset"""
    print(f"\nDownloading dataset: {dataset_name}")
    
    try:
        # Create datasets directory
        datasets_dir = Path('datasets')
        datasets_dir.mkdir(exist_ok=True)
        
        # Download dataset
        result = subprocess.run([
            'kaggle', 'datasets', 'download', '-d', dataset_name, '-p', str(datasets_dir)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded {dataset_name}")
            
            # Extract if it's a zip file
            zip_file = datasets_dir / f"{dataset_name.split('/')[-1]}.zip"
            if zip_file.exists():
                print(f"Extracting {zip_file}")
                subprocess.run(['unzip', '-q', str(zip_file), '-d', str(datasets_dir)])
                print("Extraction complete")
        else:
            print(f"Error downloading: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}")

def list_alternative_eeg_sources():
    """List alternative EEG data sources"""
    print("\nAlternative EEG Data Sources:")
    print("=" * 50)
    
    sources = [
        {
            "name": "PhysioNet EEG Motor Movement/Imagery",
            "url": "https://physionet.org/content/eegmmidb/1.0.0/",
            "description": "1,500+ EEG recordings from 109 volunteers, motor imagery tasks",
            "size": "~1.5GB",
            "format": "EDF files"
        },
        {
            "name": "OpenNeuro",
            "url": "https://openneuro.org/",
            "description": "Open-science neuroinformatics database with various EEG studies",
            "size": "Variable",
            "format": "BIDS format"
        },
        {
            "name": "DEAP Dataset",
            "url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/",
            "description": "Emotion analysis using EEG, 32 participants, 40 music videos",
            "size": "~1.5GB",
            "format": "MATLAB files"
        },
        {
            "name": "BCI Competition IV (Archive)",
            "url": "http://www.bbci.de/competition/iv/",
            "description": "Motor imagery EEG, 9 subjects, 4 classes",
            "size": "~420MB",
            "format": "GDF files"
        }
    ]
    
    for i, source in enumerate(sources, 1):
        print(f"\n{i}. {source['name']}")
        print(f"   URL: {source['url']}")
        print(f"   Description: {source['description']}")
        print(f"   Size: {source['size']}")
        print(f"   Format: {source['format']}")

def create_eeg_download_script():
    """Create a script to download EEG data from various sources"""
    script_content = '''#!/usr/bin/env python3
"""
EEG Dataset Download Script
Downloads EEG datasets from various sources for hpfracc experiments
"""

import os
import requests
import zipfile
from pathlib import Path
import mne

def download_physionet_eeg():
    """Download PhysioNet EEG Motor Movement/Imagery dataset"""
    print("Downloading PhysioNet EEG dataset...")
    
    # This would require PhysioNet credentials
    print("Note: PhysioNet requires registration and credentials")
    print("Visit: https://physionet.org/content/eegmmidb/1.0.0/")
    
def download_mne_sample():
    """Download MNE sample data for testing"""
    print("Downloading MNE sample data...")
    
    # Download sample data
    sample_data_folder = mne.datasets.sample.data_path()
    print(f"Sample data downloaded to: {sample_data_folder}")
    
    return sample_data_folder

def setup_eeg_environment():
    """Set up environment for EEG experiments"""
    print("Setting up EEG experiment environment...")
    
    # Create directories
    dirs = ['datasets', 'eeg_results', 'eeg_models']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created directory: {dir_name}")
    
    # Install required packages
    packages = [
        'mne',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'torch',
        'numpy',
        'pandas'
    ]
    
    print("Required packages:")
    for package in packages:
        print(f"  - {package}")

if __name__ == "__main__":
    print("EEG Dataset Download Script")
    print("=" * 40)
    
    # Setup environment
    setup_eeg_environment()
    
    # Download sample data
    sample_path = download_mne_sample()
    
    print("\\nSetup complete!")
    print("Ready to run EEG experiments with sample data.")
    print("For real BCI data, download from PhysioNet or other sources.")
'''
    
    with open('download_eeg_data.py', 'w') as f:
        f.write(script_content)
    
    print("\nCreated download_eeg_data.py script")
    print("Run: python download_eeg_data.py")

def main():
    """Main function"""
    print("Kaggle EEG Dataset Setup for hpfracc")
    print("=" * 50)
    
    # Setup Kaggle API
    kaggle_dir = setup_kaggle_api()
    
    # Search for EEG datasets
    search_eeg_datasets()
    
    # List alternative sources
    list_alternative_eeg_sources()
    
    # Create download script
    create_eeg_download_script()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Set up Kaggle API credentials (optional)")
    print("2. Run: python download_eeg_data.py")
    print("3. Run: python eeg_experiments.py")
    print("\nThis will give us real EEG data for honest manuscript results!")

if __name__ == "__main__":
    main()
