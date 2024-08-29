import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

class SingletonMeta(type):
    """
    A metaclass that creates a Singleton class.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # Check if an instance already exists
        if cls not in cls._instances:
            # If not, create one and store it in the _instances dictionary
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class PhiTransformer(metaclass=SingletonMeta):
    def __init__(self, model_id=None) -> None:
        self.model_id = model_id if model_id else Config.DEFAULT_MODEL
        
    def download_phi_3(self):
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype='auto')
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            local_name = self.model_id.split("/")[-1]
            local_path = os.path.join(os.getcwd(), f"models/{local_name}")
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            return {"message": "Download complete"}
        except Exception as error:
            print(f"Downloading phi3 error ~ {error}")
            return {"message": f"Downloading phi3 error ~ {error}"}
    
    
