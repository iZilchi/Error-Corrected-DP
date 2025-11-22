# setup_skin_cancer.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def setup_ham10000():
    """Setup HAM10000 dataset for federated learning - FIXED VERSION"""
    
    data_dir = './data/skin_cancer'
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    print("Files in data directory:")
    for file in os.listdir(data_dir):
        print(f"  - {file}")
    
    # Use the metadata file
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    
    if not os.path.exists(metadata_path):
        print("❌ HAM10000_metadata.csv not found!")
        return
    
    print(f"📁 Found metadata file: {metadata_path}")
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    print(f"📊 Dataset shape: {df.shape}")
    print("📋 Columns:", df.columns.tolist())
    
    # Check available image directories
    image_dirs = []
    possible_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'images']
    
    for dir_name in possible_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            image_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"🖼️ Found {image_count} images in {dir_name}")
            image_dirs.append(dir_path)
    
    if not image_dirs:
        print("❌ No image directories found!")
        return
    
    # Create train/test split
    print("🎯 Creating train/test split...")
    
    # Clean image_id - remove extensions if present
    df['image_id'] = df['image_id'].astype(str)
    df['image_id_clean'] = df['image_id'].apply(lambda x: x.replace('.jpg', ''))
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)
    
    # Save splits
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    # Create a mapping file to help the data loader find images
    create_image_mapping(data_dir, image_dirs)
    
    print("✅ Dataset prepared successfully!")
    print(f"📚 Training samples: {len(train_df)}")
    print(f"🧪 Test samples: {len(test_df)}")
    print("\n📊 Class distribution in training set:")
    print(train_df['dx'].value_counts())
    
    print("\n🎉 Setup complete! You can now run your experiments.")
    print("👉 Run: python experiments/run_comparison.py")

def create_image_mapping(data_dir, image_dirs):
    """Create a file that maps image IDs to their actual locations"""
    mapping = {}
    
    for img_dir in image_dirs:
        for file in os.listdir(img_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_id = file.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                mapping[img_id] = os.path.join(img_dir, file)
    
    # Save mapping
    import json
    with open(os.path.join(data_dir, 'image_mapping.json'), 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"📝 Created image mapping with {len(mapping)} entries")

if __name__ == "__main__":
    setup_ham10000()