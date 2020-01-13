from pyzsl.data.cub.src.core import load_model_headless


class Config:
    image_size   = (224, 224)
    model_loader = load_model_headless
    normalize    = True
    device       = 'cpu'
