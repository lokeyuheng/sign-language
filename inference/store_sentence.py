import json

def save_sentence(sign_sequence, sentence, file_path="predicted_sentences.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "sign_sequence": sign_sequence,
        "sentence": sentence
    })

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
