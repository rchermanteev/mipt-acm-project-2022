class ManagerModel:
    def __new__(cls, models):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ManagerModel, cls).__new__(cls)

        return cls.instance

    def __init__(self, models: dict):
        self._models = models

    def get_model_instance(self, name):
        return self._models[name]
