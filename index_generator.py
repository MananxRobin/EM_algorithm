# index_generator.py - Manan Ambaliya(121118776)
import json

def load_index_to_word(json_file):
    with open(json_file, 'r') as f:
        idx2word = json.load(f)
    return idx2word

if __name__ == "__main__":
    idx2word = load_index_to_word("4_map_index_to_word.json")
    print("Loaded mapping with", len(idx2word), "words.")