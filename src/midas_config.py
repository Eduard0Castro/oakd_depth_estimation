import torch
from enum import Enum

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_Hybrid = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class Midas():

    def __init__(self, model_type: ModelType = ModelType.MIDAS_SMALL) -> None:

        self.model_type = model_type.value
        try:
            self.midas = torch.hub.load('intel-isl/MiDaS', self.model_type)
        except Exception as e:
            print("Erro ao carregar modelo MiDaS: %s", e)
            raise e
        
    def device_config(self)-> None:
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")

        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.midas.to(self.device)
        self.midas.eval()

    def transform_config(self) -> None:

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
