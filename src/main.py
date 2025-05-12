import cv2
import numpy as np
import depthai as dai
from mirela_sdk.image_processing.camera.oakd_cam import OakdCam
from midas_config import Midas
import torch


def initial_config() -> OakdCam:

    oakd = OakdCam()
    stereo = oakd.get_stereo_depth()
    oakd.post_processing_stereo_depth(stereo)
    oakd.setup_camera(1)

    oakd.configure_stereo_node_output(["disparity", "rectifiedLeft", "rectifiedRight"])
    oakd.init_cam()

    return oakd, stereo

def setQueues(oakd: OakdCam, stereo: dai.node.StereoDepth) -> tuple[dai.DataOutputQueue,]:

    disp_queue = oakd.getQueue("disparity", maxSize = 1, blocking = False)
    oakd.getQueue("rectifiedLeft",  maxSize=1, blocking=False)
    oakd.getQueue("rectifiedRight", maxSize=1, blocking=False)
    rgbQueue = oakd.getQueue("rgb", 1, False)

    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    return disp_queue, rgbQueue, disparityMultiplier


if __name__ == "__main__":

    oakd, stereo = initial_config()
    disp_queue, rgbQueue, disparityMultiplier = setQueues(oakd, stereo)
    kernel = np.ones((3, 3), np.uint8)

    midas_config = Midas()
    midas_config.device_config()
    midas_config.transform_config()

    while cv2.waitKey(1) & 0xFF != ord("q"):

        disp = oakd.getFrame(disp_queue)
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

            depth_colormap = prediction.cpu().numpy()
            depth_colormap = cv2.normalize(depth_colormap, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
            depth_colormap = cv2.resize(depth_colormap, (disp.shape[1]//2, disp.shape[0]//2))

        except Exception as e:  
            print("Erro na estimativa de profundidade: %s", e)
            depth_colormap = np.zeros_like(rgb_image)

        rgb_image = cv2.resize(rgb_image, (disp.shape[1]//2, disp.shape[0]//2))
        disp = cv2.resize(disp, (disp.shape[1]//2, disp.shape[0]//2))

        disp = (disp * disparityMultiplier).astype(np.uint8)
        disp = cv2.medianBlur(disp, 3)
        disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        
        cv2.imshow("RGB original image", rgb_image)
        cv2.imshow("OAK-D depth estimation", disp)
        cv2.imshow("MiDas depth estimation", depth_colormap)

    cv2.destroyAllWindows()
    oakd.clean()