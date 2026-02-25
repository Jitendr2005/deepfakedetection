"""
Download image datasets from HuggingFace
"""
from pathlib import Path
from config import DATASET_CONFIG

def download_huggingface_dataset(dataset_name: str, download_path: Path):
    """Download image dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        print(f"Downloading {dataset_name}...")
        download_path.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(str(download_path))
        print(f"✓ Downloaded to {download_path}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"\nTip: Some datasets may require authentication.")
        print(f"Try: huggingface-cli login")
        return False

def main():
    """Main download function"""
    import sys
    
    print("Deepfake Detection Dataset Downloader")
    print("="*50)
    print("\nAvailable datasets:")
    print("1. Deepfake Images (210k): JamieWithofs/Deepfake-and-real-images-4")
    print("2. Deepfake Images (190k): Hemg/deepfake-and-real-images")
    print("3. Deepfake Images (122k): JamieWithofs/Deepfake-and-real-images")
    print("\nUsage:")
    print("  python3 download_datasets.py --dataset JamieWithofs/Deepfake-and-real-images-4")
    print("  python3 download_datasets.py --dataset Hemg/deepfake-and-real-images")
    
    if len(sys.argv) > 1 and "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            dataset_name = sys.argv[idx + 1]
            download_path = DATASET_CONFIG["huggingface"]["faceforensics"]["path"]
            download_huggingface_dataset(dataset_name, download_path)
    elif len(sys.argv) > 1 and "--huggingface" in sys.argv:
        # Legacy support
        idx = sys.argv.index("--huggingface")
        if idx + 1 < len(sys.argv):
            dataset = sys.argv[idx + 1]
            if dataset == "faceforensics":
                print("\nNote: Using alternative dataset: JamieWithofs/Deepfake-and-real-images-4")
                download_huggingface_dataset(
                    "JamieWithofs/Deepfake-and-real-images-4",
                    DATASET_CONFIG["huggingface"]["faceforensics"]["path"]
                )
            elif dataset == "wilddeepfake":
                print("\nNote: Using alternative dataset: Hemg/deepfake-and-real-images")
                download_huggingface_dataset(
                    "Hemg/deepfake-and-real-images",
                    DATASET_CONFIG["huggingface"]["wilddeepfake"]["path"]
                )

if __name__ == "__main__":
    main()
