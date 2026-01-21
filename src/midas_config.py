import torch
from enum import Enum

from collections import deque
import numpy as np


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


class Calibrate:
        
    # Model: z_oak ≈ a/(midas + eps) + b
    EPS = 1e-6

    MIDAS_SAMPLES = deque(maxlen=600)
    OAKD_SAMPLES  = deque(maxlen=600)

    A_INV = None
    B_INV = None

    MIN_SAMPLES_TO_CALIBRATE = 40      
    RECALIBRATE_N            = 20          
    NEW_SAMPLES_SINCE_CALIB  =  0      


    @classmethod
    def fit_inverse_calibration(cls, eps: float = 1e-6):

        """
        Adjustment: z_oak = a * (1/(midas+eps)) + b
        
        :param eps(float): numerical stability
        
        Returns (a, b) using least squares through the polyfit method.
         
        """


        midas_arr = np.asarray(Calibrate.MIDAS_SAMPLES, dtype=np.float64)
        oak_arr   = np.asarray(Calibrate.OAKD_SAMPLES, dtype=np.float64)

        valid = np.isfinite(midas_arr) & np.isfinite(oak_arr) & (midas_arr > 0) & (oak_arr > 0)
        if np.count_nonzero(valid) < cls.MIN_SAMPLES_TO_CALIBRATE:
            return None, None

        x = 1.0 / (midas_arr[valid] + eps)
        y = oak_arr[valid]

        a, b = np.polyfit(x, y, 1)  # y = a*x + b
        return float(a), float(b)


    @classmethod
    def midas_to_meters(cls, depth_midas_mean, eps=1e-6):

        """
        Converts MiDaS average (relative) to meters (estimated), using inverse calibration.
        
        """

        if cls.A_INV is None or cls.B_INV is None:
            return float("nan")
        if not np.isfinite(depth_midas_mean) or depth_midas_mean <= 0:
            return float("nan")
        return float(cls.A_INV / (depth_midas_mean + eps) + cls.B_INV)
