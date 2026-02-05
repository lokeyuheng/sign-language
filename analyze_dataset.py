from dataset.download_wlasl import load_wlasl
from collections import Counter
import json

def analyze_dataset():
    print("Loading WLASL for analysis...")
    dataset = load_wlasl(split="train")
    
    # Extract labels and glosses
    label_gloss_map = {}
    labels = []
    
    for item in dataset:
        l = item["label"]
        g = item.get("gloss", f"sign_{l}")
        labels.append(l)
        label_gloss_map[l] = g
        
    counts = Counter(labels)
    top_100 = counts.most_common(100)
    
    print(f"\nTotal Samples: {len(labels)}")
    print(f"Total Classes: {len(counts)}")
    print("\nTop 10 Signs:")
    for l, count in top_100[:10]:
        print(f"  {label_gloss_map[l]} (ID {l}): {count} samples")
        
    # Save top 100 IDs
    top_100_ids = [l for l, count in top_100]
    with open("dataset/top_100_signs.json", "w") as f:
        json.dump({"top_100": top_100_ids, "map": {l: label_gloss_map[l] for l in top_100_ids}}, f, indent=2)
    
    print("\nSaved top 100 sign IDs to dataset/top_100_signs.json")

if __name__ == "__main__":
    analyze_dataset()
