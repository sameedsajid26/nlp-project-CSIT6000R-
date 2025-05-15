# models/__init__.py
from models.gpt2_model import GPT2Model
from models.t5_model import T5Model

# Factory function to get model by name
def get_model(model_type, model_name=None):
    if model_type.lower() == "gpt2":
        return GPT2Model(model_name or "gpt2")
    elif model_type.lower() == "t5":
        return T5Model(model_name or "google/flan-t5-small")
    else:
        raise ValueError(f"Unknown model type: {model_type}")