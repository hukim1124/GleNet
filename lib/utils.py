import json

def load_json(path):
    with open(path, 'rt') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'wt') as f:
        json.dump(data, f, indent=4)
