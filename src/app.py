import cv2
import depthai as dai

from mirela_sdk.image_processing.camera.oakd_cam import OakdCam
from midas_config import Midas, Calibrate

import torch
import numpy as np
from typing import Tuple


def initial_config() -> Tuple[OakdCam, dai.node.StereoDepth]:

    oakd = OakdCam()
    stereo = oakd.depth_config(link_out=False)
    stereo.setDepthAlign(oakd.oak_dict.get(1)[1]) # alinhas o mapa de profundidade com a CAM_A (rgb)
    stereo.setOutputSize(960, 540)
    # oakd.configure_stereo_node_output(["disparity"])
    
    oakd.setup_camera(1, link_out = False)

    oakd.create_sync_node(stereo)
    # oakd.post_processing_stereo_depth(stereo)
    oakd.init_cam(usb2_mode = True)

    return oakd, stereo


def setQueues(oakd: OakdCam) -> tuple[dai.DataOutputQueue, dai.DataOutputQueue]:

    disparity_queue = None
    # disparity_queue = oakd.getQueue("disparity", maxSize = 1, blocking = False)
    grp_queue = oakd.getQueue("xout_grp", 1, False)

    return grp_queue, disparity_queue


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


def calibrate(target_frame_mean: float, depth_midas_mean: float, depth: bool = True) -> float:

    calibration = Calibrate.fit_inverse_calibration if depth else Calibrate.fit_direct_calibration

    if np.isfinite(target_frame_mean) and target_frame_mean > 0 and np.isfinite(depth_midas_mean) \
           and depth_midas_mean > 0:
            
            Calibrate.OAKD_SAMPLES.append(target_frame_mean)
            Calibrate.MIDAS_SAMPLES.append(depth_midas_mean)
            Calibrate.NEW_SAMPLES_SINCE_CALIB += 1

    if Calibrate.A_INV is None or Calibrate.B_INV is None:
        if len(Calibrate.OAKD_SAMPLES) >= Calibrate.MIN_SAMPLES_TO_CALIBRATE:
            Calibrate.A_INV, Calibrate.B_INV = calibration()
            Calibrate.NEW_SAMPLES_SINCE_CALIB = 0


    else:
        if Calibrate.NEW_SAMPLES_SINCE_CALIB >= Calibrate.RECALIBRATE_N:
            a_new, b_new = calibration()
            Calibrate.NEW_SAMPLES_SINCE_CALIB = 0
            if a_new is not None: Calibrate.A_INV, Calibrate.B_INV = a_new, b_new


    return Calibrate.midas_to_meters(depth_midas_mean)


if __name__ == "__main__":

    oakd, stereo = initial_config()
    grp_queue, disparity_queue = setQueues(oakd)
    kernel = np.ones((3, 3), np.uint8)

    midas_config = Midas()
    midas_config.device_config()
    midas_config.transform_config()

    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()


    while cv2.waitKey(1) & 0xFF != ord("q"):

        group_msg = grp_queue.get()
        depth_oakd = group_msg["depth"].getFrame()
        rgb_image = group_msg["rgb"].getCvFrame()
        
        # disparity_oakd = oakd.getFrame(disparity_queue)  

        # Faço o cálculo de pixels inválidos antes do resize e de qualquer normalização
        invalid_pixels = ~np.isfinite(depth_oakd) | (depth_oakd <= 0)
        valid_pixels = ~invalid_pixels
        invalid_percent = np.mean(invalid_pixels)*100;
        # print("Invalid pixels before: ", invalid_percent)


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


        # _, _, _, disparity_frame_mean = calculate_roi(disparity_oakd)
        # OAK-D stereo depth frame:
        width_interested, height_interested, roi_area, depth_oakd_mean = calculate_roi(depth_oakd)
        # MiDaS depth frame:
        midas_width_interested, midas_height_interested, midas_roi_area, depth_midas_mean = calculate_roi(depth_midas)
        depth_midas_m = calibrate(depth_oakd_mean, depth_midas_mean, True)


        if Calibrate.A_INV and Calibrate.B_INV:

            Calibrate.RMS_SAMPLES_COUNT += 1
            Calibrate.SQUARE_ERROR_LIST.append((depth_oakd_mean - depth_midas_m)**2)

            if Calibrate.RMS_SAMPLES_COUNT == Calibrate.RMS_MAX_SAMPLES:
                Calibrate.RMS_SAMPLES_COUNT = 0
                rms_error = np.sqrt(np.mean(Calibrate.SQUARE_ERROR_LIST))
                print("RMS Error: ", rms_error)
                if rms_error <= Calibrate.RMS_ERROR_THRESHOLD:
                    depth_oakd[invalid_pixels] = 1*Calibrate.A_INV / (depth_midas[invalid_pixels] + Calibrate.EPS + Calibrate.B_INV)
                    invalid_pixels = ~np.isfinite(depth_oakd) | (depth_oakd <= 0)
                    # invalid_percent = np.mean(invalid_pixels)*100; print("Invalid pixels after: ", invalid_percent)

                Calibrate.SQUARE_ERROR_LIST.clear()

        depth_midas = cv2.normalize(depth_midas, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8)
        depth_oakd = cv2.normalize(depth_oakd, None, 0, 255, 
                                cv2.NORM_MINMAX).astype(np.uint8)

        depth_oakd = cv2.normalize(depth_oakd, None, 0, 255, 
                                cv2.NORM_MINMAX).astype(np.uint8)
        depth_oakd = cv2.medianBlur(depth_oakd, 3)
        depth_oakd = cv2.morphologyEx(depth_oakd, cv2.MORPH_CLOSE, kernel)
        depth_oakd = cv2.applyColorMap(depth_oakd, cv2.COLORMAP_JET)
        depth_oakd = cv2.rectangle(depth_oakd, (width_interested, height_interested), 
                                    (width_interested + roi_area[0], height_interested + roi_area[1]), 
                                    (255, 255, 255), 1)
        depth_oakd = cv2.putText(depth_oakd, f"z: {depth_oakd_mean:.3f}",
                                  (width_interested + 10, height_interested + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



        depth_midas = cv2.applyColorMap(depth_midas, cv2.COLORMAP_JET)
        depth_midas = cv2.rectangle(depth_midas, (midas_width_interested, midas_height_interested), 
                                    (midas_width_interested + midas_roi_area[0], midas_height_interested + midas_roi_area[1]), 
                                    (255, 255, 255), 1)
        
        depth_midas = cv2.putText(depth_midas, f"z: {depth_midas_m:.3f}", 
                                  (width_interested + 10, height_interested + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        # disparity_oakd = (disparity_oakd * disparityMultiplier).astype(np.uint8)

        # #Aplica um mapeamento dos valores de disparidade em diferentes cores para a visualização
        # #Vai de azul (distante) a vermelho (perto)
        # disparity_oakd = cv2.applyColorMap(disparity_oakd, cv2.COLORMAP_JET)

        cv2.imshow("OAK-D depth estimation", depth_oakd)
        cv2.imshow("RGB Image", rgb_image)
        cv2.imshow("MiDas depth estimation", depth_midas)

    cv2.destroyAllWindows()
    oakd.close()
