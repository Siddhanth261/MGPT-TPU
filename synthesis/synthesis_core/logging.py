import json, time

class JsonLogger:
    def __init__(self, path):
        self.f = open(path, "a")

    def log(self, d):
        d["t"] = time.time()
        self.f.write(json.dumps(d) + "\n")
