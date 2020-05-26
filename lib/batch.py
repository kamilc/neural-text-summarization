class Batch:
    def __init__(self, data, ix=0):
        self.data = data
        self.ix = ix


    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(f"Attribute missing: {name}")

