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
    oakd.setup_camera(1)
    oakd.init_cam()

    return oakd, stereo


def setQueues(oakd: OakdCam) -> tuple[dai.DataOutputQueue, dai.DataOutputQueue]:

    depth_queue = oakd.getQueue("depth", maxSize = 1, blocking = False)
    rgbQueue = oakd.getQueue("rgb", 1, False)

    return depth_queue, rgbQueue


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


def init_depth_plot(buffer_size=100):

    """
    Initializes a real-time plot for depth comparison + running MSE.
    Returns a dict containing the plot state.
    """
    
    plt.ion()

    state = {}
    state["x"] = np.arange(buffer_size)
    state["oakd"] = [np.nan] * buffer_size
    state["midas"] = [np.nan] * buffer_size
    state["mse"] = [np.nan] * buffer_size

    state["fig"], state["ax"] = plt.subplots()

    state["line_oakd"], = state["ax"].plot(state["x"], state["oakd"], label="OAK-D (meters)")
    state["line_midas"], = state["ax"].plot(state["x"], state["midas"], label="MiDaS calibrated (meters)")

    state["ax"].set_ylim(0, 5)
    state["ax"].set_xlim(0, buffer_size)
    state["ax"].set_xlabel("Time")
    state["ax"].set_ylabel("Depth (m)")
    state["ax"].grid(True)

    state["ax2"] = state["ax"].twinx()
    state["line_mse"], = state["ax2"].plot(state["x"], state["mse"], label="MSE (m²)", color = "red")
    state["ax2"].set_ylabel("MSE (m²)")

    lines = [state["line_oakd"], state["line_midas"], state["line_mse"]]
    labels = [l.get_label() for l in lines]
    state["ax"].legend(lines, labels, loc="upper right")

    state["sse"] = 0.0
    state["n"] = 0

    return state


def update_depth_plot(depth_frame_mean, depth_midas_m, state):
    """
    Updates the real-time depth plot + running MSE.
    """
    if np.isfinite(depth_frame_mean) and np.isfinite(depth_midas_m):
        depth_frame_mean = float(depth_frame_mean)
        depth_midas_m = float(depth_midas_m)

        err = depth_midas_m - depth_frame_mean
        state["sse"] += err * err
        state["n"] += 1
        mse_running = state["sse"] / max(state["n"], 1)

        state["oakd"].pop(0)
        state["midas"].pop(0)
        state["mse"].pop(0)

        state["oakd"].append(depth_frame_mean)
        state["midas"].append(depth_midas_m)
        state["mse"].append(mse_running)

        state["line_oakd"].set_ydata(state["oakd"])
        state["line_midas"].set_ydata(state["midas"])
        state["line_mse"].set_ydata(state["mse"])

        state["ax"].relim()
        state["ax"].autoscale_view(True, True, True)
        state["ax2"].relim()
        state["ax2"].autoscale_view(True, True, True)

        state["fig"].canvas.draw_idle()
        state["fig"].canvas.flush_events()
        plt.pause(0.001)


def calibrate(depth_frame_mean: float, depth_midas_mean: float) -> float:

    if np.isfinite(depth_frame_mean) and depth_frame_mean > 0 and np.isfinite(depth_midas_mean) \
           and depth_midas_mean > 0:
            
            Calibrate.OAKD_SAMPLES.append(depth_frame_mean)
            Calibrate.MIDAS_SAMPLES.append(depth_midas_mean)
            Calibrate.NEW_SAMPLES_SINCE_CALIB += 1

    if Calibrate.A_INV is None or Calibrate.B_INV is None:
        if len(Calibrate.OAKD_SAMPLES) >= Calibrate.MIN_SAMPLES_TO_CALIBRATE:
            Calibrate.A_INV, Calibrate.B_INV = Calibrate.fit_inverse_calibration(eps=eps)
            Calibrate.NEW_SAMPLES_SINCE_CALIB = 0
            if Calibrate.A_INV is not None:
                print(f"[CALIB INIT] Calibrate.A_INV={Calibrate.A_INV:.6f} \
                      Calibrate.B_INV={Calibrate.B_INV:.6f}  n={len(Calibrate.OAKD_SAMPLES)}")

    else:
        if Calibrate.NEW_SAMPLES_SINCE_CALIB >= Calibrate.RECALIBRATE_N:
            a_new, b_new = Calibrate.fit_inverse_calibration(eps=eps)
            Calibrate.NEW_SAMPLES_SINCE_CALIB = 0
            if a_new is not None:
                Calibrate.A_INV, Calibrate.B_INV = a_new, b_new
                print(f"[CALIB UPDATE] Calibrate.A_INV={Calibrate.A_INV:.6f}\
                      Calibrate.B_INV={Calibrate.B_INV:.6f}  n={len(Calibrate.OAKD_SAMPLES)}")

    return Calibrate.midas_to_meters(depth_midas_mean, eps=eps)


if __name__ == "__main__":

    oakd, _ = initial_config()
    depth_queue, rgbQueue = setQueues(oakd)
    kernel = np.ones((3, 3), np.uint8)

    midas_config = Midas()
    midas_config.device_config()
    midas_config.transform_config()
    eps = 1e-6

    plot_state = init_depth_plot(buffer_size=100)

    frame_count = 0
    plot_every_n = 2

    while cv2.waitKey(1) & 0xFF != ord("q"):

        depth_oakd = oakd.getFrame(depth_queue)     

        if depth_oakd.dtype != np.float32:
            depth_oakd = depth_oakd.astype(np.float32) / 1000.0

        rgb_image = oakd.getFrame(rgbQueue)

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

        depth_oakd = cv2.normalize(depth_oakd, None, 0, 255, 
                                cv2.NORM_MINMAX).astype(np.uint8)
        depth_oakd = cv2.medianBlur(depth_oakd, 3)
        depth_oakd = cv2.morphologyEx(depth_oakd, cv2.MORPH_CLOSE, kernel)
        depth_oakd = cv2.applyColorMap(depth_oakd, cv2.COLORMAP_JET)
        depth_oakd = cv2.rectangle(depth_oakd, (width_interested, height_interested), 
                                    (width_interested + roi_area[0], height_interested + roi_area[1]), 
                                    (255, 255, 255), 1)
        depth_oakd = cv2.putText(depth_oakd, f"z: {depth_frame_mean:.3f}", 
                                  (width_interested + 10, height_interested + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # MiDaS depth frame:
        width_interested, height_interested, roi_area, depth_midas_mean = calculate_roi(depth_midas)

        depth_midas_m = calibrate(depth_frame_mean, depth_midas_mean)

        depth_midas = cv2.normalize(depth_midas, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8)
        depth_midas = cv2.applyColorMap(depth_midas, cv2.COLORMAP_JET)
        depth_midas = cv2.rectangle(depth_midas, (width_interested, height_interested), 
                                    (width_interested + roi_area[0], height_interested + roi_area[1]), 
                                    (255, 255, 255), 1)
        depth_midas = cv2.putText(depth_midas, f"z: {depth_midas_m:.3f}", 
                                  (width_interested + 10, height_interested + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame_count += 1
        if frame_count % plot_every_n == 0:
            update_depth_plot(depth_frame_mean, depth_midas_m, plot_state)

        cv2.imshow("RGB   original image",     rgb_image)
        cv2.imshow("OAK-D depth estimation", depth_oakd)
        cv2.imshow("MiDas depth estimation", depth_midas)

    try:
        plt.ioff()
        plt.close("all")
    except Exception:
        pass

    cv2.destroyAllWindows()
    oakd.close()
