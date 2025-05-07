import pickle, os

def save_ckpt(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_ckpt(path):
    with open(path, "rb") as f:
        return pickle.load(f)
