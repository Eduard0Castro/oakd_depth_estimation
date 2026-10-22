import cv2
import depthai as dai

from mirela_sdk.image_processing.camera.oakd_cam import OakdCam
from midas_config import Midas, Calibrate

import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def initial_config() -> Tuple[OakdCam, dai.node.StereoDepth]:

    oakd = OakdCam()
    stereo = oakd.depth_config()
    stereo.setDepthAlign(oakd.oak_dict.get(1)[1]) # alinhas o mapa de profundidade com a CAM_A (rgb)
    stereo.setOutputSize(960, 540)
    oakd.configure_stereo_node_output(["disparity"])
    
    oakd.setup_camera(1)
    oakd.create_sync_node(stereo)
    oakd.init_cam(usb2_mode = True)

    return oakd, stereo


def setQueues(oakd: OakdCam) -> tuple[dai.DataOutputQueue, dai.DataOutputQueue]:

    # depth_queue = oakd.getQueue("depth", maxSize = 1, blocking = False)
    # disparity_queue = oakd.getQueue("disparity", maxSize = 1, blocking = False)
    # rgbQueue = oakd.getQueue("rgb", 1, False)

    grp_queue = oakd.getQueue("xout_grp", 1, False)

    return grp_queue


def calculate_roi(frame: np.array) -> Tuple[int, int, Tuple[int, int], float]:

    width_interested  = frame.shape[1]//2 - int((frame.shape[1] / 640) * 100)
    height_interested = frame.shape[0]//2 - int((frame.shape[0] / 400) * 100)

    roi_area = (int((frame.shape[1] / 640) *200), int((frame.shape[0] / 400) *200))
    
    width_interested  = max(0, min(width_interested,  frame.shape[1] - roi_area[0]))
    height_interested = max(0, min(height_interested, frame.shape[0] - roi_area[1]))
    
    roi_depth_frame = frame[height_interested : height_interested + roi_area[1], 
                            width_interested  : width_interested  + roi_area[0]]

    valid = np.isfinite(roi_depth_frame)
    valid &= (roi_depth_frame > 0)

    frame_mean = float(np.mean(roi_depth_frame[valid])) if np.any(valid) else float("nan")
    
    return width_interested, height_interested, roi_area, frame_mean


if __name__ == "__main__":

    oakd, stereo = initial_config()
    grp_queue= setQueues(oakd)
    kernel = np.ones((3, 3), np.uint8)

    midas_config = Midas()
    midas_config.device_config()
    midas_config.transform_config()
    eps = 1e-6

    while cv2.waitKey(1) & 0xFF != ord("q"):

        group_msg = grp_queue.get()
        
        depth_oakd = group_msg["depth"].getFrame()
        rgb_image = group_msg["rgb"].getCvFrame()
        
        # disparity_oakd = oakd.getFrame(disparity_queue)  

        # Faço o cálculo de pixels inválidos antes do resize e de qualquer normalização
        invalid_pixels = ~np.isfinite(depth_oakd) | (depth_oakd <= 0)
        invalid_percent = np.mean(invalid_pixels)*100;print("Invalid percent: ", invalid_percent)


        if depth_oakd.dtype != np.float32:
            depth_oakd = depth_oakd.astype(np.float32) / 1000.0

        try:
            img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            input_tensor = midas_config.transform(img_rgb).to(midas_config.device)

            with torch.no_grad():
                prediction = midas_config.midas(input_tensor)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_midas = prediction.cpu().numpy().astype(np.float32)

        except Exception as e:  
            print("Error in depth estimation: %s", e)
            depth_midas = np.zeros(rgb_image.shape[:2], dtype=np.float32)

        # OAK-D stereo depth frame:
        width_interested, height_interested, roi_area, depth_frame_mean = calculate_roi(depth_oakd)

        invalid_pixels = ~np.isfinite(depth_oakd) | (depth_oakd <= 0)
        # depth_oakd[invalid_pixels] = depth_midas[invalid_pixels]

        depth_midas = cv2.normalize(depth_midas, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8)
        depth_oakd = cv2.normalize(depth_oakd, None, 0, 255, 
                                cv2.NORM_MINMAX).astype(np.uint8)



        disparity_oakd = depth_oakd
        disparity_oakd = cv2.normalize(disparity_oakd, None, 0, 255, 
                                cv2.NORM_MINMAX).astype(np.uint8)
        disparity_oakd = cv2.medianBlur(disparity_oakd, 3)
        disparity_oakd = cv2.morphologyEx(disparity_oakd, cv2.MORPH_CLOSE, kernel)
        disparity_oakd = cv2.applyColorMap(disparity_oakd, cv2.COLORMAP_JET)
        disparity_oakd = cv2.rectangle(disparity_oakd, (width_interested, height_interested), 
                                    (width_interested + roi_area[0], height_interested + roi_area[1]), 
                                    (255, 255, 255), 1)
        disparity_oakd = cv2.putText(disparity_oakd, f"z: {depth_frame_mean:.3f}", 
                                  (width_interested + 10, height_interested + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # MiDaS depth frame:
        # width_interested, height_interested, roi_area, depth_midas_mean = calculate_roi(depth_midas)

        depth_midas = cv2.applyColorMap(depth_midas, cv2.COLORMAP_JET)
        depth_midas = cv2.rectangle(depth_midas, (width_interested, height_interested), 
                                    (width_interested + roi_area[0], height_interested + roi_area[1]), 
                                    (255, 255, 255), 1)
        # depth_midas = cv2.putText(depth_midas, f"z: {depth_midas_mean:.3f}", 
        #                           (width_interested + 10, height_interested + 15), 
        #                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        cv2.imshow("OAK-D depth estimation", disparity_oakd)
        cv2.imshow("RGB Image", rgb_image)
        cv2.imshow("MiDas depth estimation", depth_midas)

    cv2.destroyAllWindows()
    oakd.close()
