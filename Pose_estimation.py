import argparse
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


fps_time = 0


'''
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default=0, help='video path or cam number')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=4.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    
    parser.add_argument('--show', type=bool, default=False)

    '''
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    '''
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=False, show=args.show)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=False, show=args.show)
    cam = cv2.VideoCapture(args.video)
    ret_val, image = cam.read()

    #Save video
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    if type(args.video) == int:
        out = cv2.VideoWriter('output/output_live.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    else:
        out = cv2.VideoWriter('output/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        out.write(image)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()
