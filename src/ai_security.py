#!/usr/bin/env python3
"""
ai_security.py

Refactored inference script for Raspberry Pi (CSI / USB camera) using ONNX Runtime.
GUI-friendly:
- --headless : no cv2 windows
- --json     : prints single AI_RESULT_JSON=... line

IMPORTANT:
Picamera2 output channel order can vary by config/pipeline.
Use:
  --picam-order bgr   (default)
or
  --picam-order rgb
to fix weird colors.
"""

import os
import sys
import time
import json
import argparse
from typing import Optional, List, Tuple

import numpy as np
import cv2
import onnxruntime as ort
import subprocess

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

try:
    from gpiozero import Button
    GPIOZERO_AVAILABLE = True
except Exception:
    GPIOZERO_AVAILABLE = False


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr: np.ndarray, input_size=(640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    # Model wants RGB, we accept BGR then convert
    img, scale, pad = letterbox(img_bgr, new_shape=input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, 0)
    return img, scale, pad


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction: np.ndarray, conf_thres=0.25, iou_thres=0.45):
    if prediction.size == 0:
        return []
    if prediction.ndim == 3:
        prediction = prediction[0]

    xywh = prediction[:, :4]
    obj_conf = prediction[:, 4]
    class_conf = prediction[:, 5:]

    class_id = np.argmax(class_conf, axis=1)
    class_score = class_conf[np.arange(len(class_conf)), class_id]
    conf = obj_conf * class_score

    mask = conf > conf_thres
    if not np.any(mask):
        return []

    xyxy = xywh2xyxy(xywh[mask])
    conf = conf[mask]
    class_id = class_id[mask]

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = conf.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    detections = []
    for idx in keep:
        x1i, y1i, x2i, y2i = xyxy[idx]
        detections.append((x1i, y1i, x2i, y2i, float(conf[idx]), int(class_id[idx])))
    return detections


def load_builtin_coco_names():
    return [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
        'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
        'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
        'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
        'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]


def capture_with_libcamera(tmp_file='/tmp/libcamera_capture.jpg', debug=False):
    try:
        result = subprocess.run(
            ['libcamera-jpeg', '-o', tmp_file, '-t', '1', '--width', '640', '--height', '480'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode != 0:
            if debug:
                print(f'libcamera-jpeg failed: {result.stderr}', file=sys.stderr)
            return None
        return cv2.imread(tmp_file)  # BGR
    except FileNotFoundError:
        if debug:
            print('libcamera-jpeg not found. Install: sudo apt install libcamera-tools', file=sys.stderr)
        return None
    except Exception as e:
        if debug:
            print(f'libcamera capture failed: {e}', file=sys.stderr)
        return None


def scale_coords(detections: List[Tuple], scale: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]):
    scaled = []
    left, top = pad
    orig_h, orig_w = orig_shape
    for x1, y1, x2, y2, conf, cls in detections:
        x1 = (x1 - left) / scale
        x2 = (x2 - left) / scale
        y1 = (y1 - top) / scale
        y2 = (y2 - top) / scale
        x1 = max(0, min(orig_w - 1, x1))
        x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1))
        y2 = max(0, min(orig_h - 1, y2))
        scaled.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
    return scaled


def draw_detections(img: np.ndarray, detections: List[Tuple], class_names: List[str] = None):
    for (x1, y1, x2, y2, conf, cls) in detections:
        label = f"class{cls} {conf:.2f}"
        if class_names and 0 <= cls < len(class_names):
            label = f"{class_names[cls]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def create_session(model_path: str, num_threads: int = 4):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.log_severity_level = 3
    os.environ.setdefault('OMP_NUM_THREADS', str(num_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(num_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(num_threads))
    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=['CPUExecutionProvider'])


def normalize_yolo_output(pred: np.ndarray) -> np.ndarray:
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim == 2:
        if pred.shape[0] > pred.shape[1]:
            return np.expand_dims(pred, 0)
        elif pred.shape[0] == 5:
            pred = pred.T
            num_preds = pred.shape[0]
            output = np.zeros((num_preds, 85), dtype=pred.dtype)
            output[:, :4] = pred[:, :4]
            output[:, 4] = pred[:, 4]
            output[:, 5:] = pred[:, 4:5]
            return np.expand_dims(output, 0)
    if pred.ndim == 2:
        return np.expand_dims(pred, 0)
    return pred


def run_inference_frame(sess, img: np.ndarray, input_name: str):
    input_type_str = str(sess.get_inputs()[0].type)
    img_input = img.astype(np.float16) if 'float16' in input_type_str else img.astype(np.float32)

    outputs = sess.run(None, {input_name: img_input})
    if isinstance(outputs, list) and len(outputs) == 1:
        pred = outputs[0]
    elif isinstance(outputs, list):
        try:
            pred = np.concatenate([o.reshape(1, -1, o.shape[-1]) for o in outputs], axis=1)
        except Exception:
            pred = outputs[0]
    else:
        pred = outputs

    return normalize_yolo_output(np.array(pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.onnx')
    parser.add_argument('--camera', choices=['csi', 'usb'], default='usb')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gpio-pin', type=int, default=17)
    parser.add_argument('--no-wait', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input-image', type=str, default='')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--output', type=str, default='result.jpg')
    parser.add_argument('--capture-output', type=str, default='capture.jpg')
    parser.add_argument('--class-names', type=str, default='')
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--json', action='store_true')

    # ✅ NEW: choose picamera channel order
    parser.add_argument('--picam-order', choices=['bgr', 'rgb'], default='bgr',
                        help='How Picamera2 capture_array() is ordered. Use rgb if colors look wrong.')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'Model not found: {args.model}', file=sys.stderr)
        sys.exit(1)

    if args.input_image:
        if not os.path.exists(args.input_image):
            print(f'Input image not found: {args.input_image}', file=sys.stderr)
            sys.exit(1)
        args.no_wait = True

    class_names = []
    if args.class_names:
        if args.class_names == 'default':
            class_names = load_builtin_coco_names()
        elif os.path.exists(args.class_names):
            with open(args.class_names, 'r') as f:
                class_names = [x.strip() for x in f.readlines() if x.strip()]

    sess = create_session(args.model, num_threads=args.num_threads)
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    model_h, model_w = 640, 640
    if len(input_shape) >= 3 and input_shape[-2] and input_shape[-1]:
        model_h, model_w = input_shape[-2], input_shape[-1]

    cap: Optional[cv2.VideoCapture] = None
    picam2 = None
    libcamera_fallback = False

    def open_video_capture(device_idx: int, width: int, height: int):
        backends = []
        if hasattr(cv2, 'CAP_V4L2'):
            backends.append(cv2.CAP_V4L2)
        backends.append(cv2.CAP_ANY)
        if hasattr(cv2, 'CAP_GSTREAMER'):
            backends.append(cv2.CAP_GSTREAMER)

        for b in backends:
            try:
                cap_try = cv2.VideoCapture(device_idx, b)
            except Exception:
                try:
                    cap_try = cv2.VideoCapture(device_idx)
                except Exception:
                    cap_try = None

            if cap_try is None or not cap_try.isOpened():
                continue

            cap_try.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ret, _ = cap_try.read()
            if ret:
                return cap_try
            cap_try.release()
        return None

    if not args.input_image:
        if args.camera == 'csi' and PICAMERA2_AVAILABLE:
            try:
                picam2 = Picamera2()
                # Keep RGB888 (stable), we handle order with --picam-order
                cfg = {'format': 'RGB888', 'size': (model_w, model_h)}
                config = picam2.create_preview_configuration(cfg)
                picam2.configure(config)
                picam2.start()
                time.sleep(0.2)
            except Exception as e:
                if args.debug:
                    print(f'Picamera2 init failed: {e}', file=sys.stderr)
                picam2 = None
                libcamera_fallback = True

        if picam2 is None and not libcamera_fallback:
            cap = open_video_capture(args.device, model_w, model_h)
            if cap is None:
                print('Unable to open camera.', file=sys.stderr)
                sys.exit(1)

    for _ in range(3):
        dummy = np.zeros((1, 3, model_h, model_w), dtype=np.float32)
        try:
            sess.run(None, {input_name: dummy})
        except Exception:
            break

    if args.gpio_pin is not None and not args.no_wait:
        if GPIOZERO_AVAILABLE:
            trigger_button = Button(args.gpio_pin, pull_up=True, bounce_time=0.01)
            trigger_button.wait_for_press(timeout=None)
            trigger_button.close()
        else:
            print('gpiozero not available.', file=sys.stderr)
            sys.exit(1)

    best_cls_name = None
    best_conf = 0.0
    detections_count = 0
    proc_time = 0.0
    orig_w = orig_h = None

    try:
        start = time.time()

        if args.input_image:
            frame_bgr = cv2.imread(args.input_image)  # BGR
            if frame_bgr is None:
                print(f'Failed to load image: {args.input_image}', file=sys.stderr)
                sys.exit(1)

        elif picam2:
            frame = picam2.capture_array()  # could be RGB or BGR depending on pipeline
            if args.picam_order == "rgb":
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # treat capture_array as BGR already
                frame_bgr = frame

        elif libcamera_fallback:
            frame_bgr = capture_with_libcamera(debug=args.debug)  # BGR
            if frame_bgr is None:
                print('libcamera-jpeg capture failed.', file=sys.stderr)
                sys.exit(1)

        else:
            ret, frame_bgr = cap.read()  # BGR
            if not ret or frame_bgr is None:
                print('Frame grab failed.', file=sys.stderr)
                sys.exit(1)

        orig_h, orig_w = frame_bgr.shape[:2]

        img, scale, pad = preprocess(frame_bgr, input_size=(model_h, model_w))
        pred = run_inference_frame(sess, img, input_name)

        if not class_names:
            try:
                arr = np.array(pred)
                num_cls = arr.shape[-1] - 5
                if num_cls == 80:
                    class_names = load_builtin_coco_names()
            except Exception:
                pass

        detections = non_max_suppression(pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        detections = scale_coords(detections, scale, pad, (orig_h, orig_w))
        detections_count = len(detections)

        if detections:
            x1, y1, x2, y2, conf, cls = max(detections, key=lambda d: d[4])
            best_conf = float(conf)
            if class_names and 0 <= cls < len(class_names):
                best_cls_name = class_names[cls]
            else:
                best_cls_name = f"class{cls}"

        # Save + draw (OpenCV expects BGR)
        cv2.imwrite(args.capture_output, frame_bgr)
        out = draw_detections(frame_bgr.copy(), detections, class_names)

        end = time.time()
        proc_time = float(end - start)

        cv2.putText(out, f"Time: {proc_time:.3f}s", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(args.output, out)

        if args.json:
            payload = {
                "ok": True,
                "class": best_cls_name,
                "confidence": best_conf,
                "detections": detections_count,
                "processing_time": proc_time,
                "resolution": [int(orig_w), int(orig_h)] if orig_w and orig_h else None,
                "capture_path": os.path.abspath(args.capture_output),
                "result_path": os.path.abspath(args.output),
                "picam_order": args.picam_order,
            }
            print("AI_RESULT_JSON=" + json.dumps(payload))

        if not args.headless:
            cv2.imshow('capture', cv2.imread(args.capture_output))
            cv2.imshow('result', out)
            while True:
                try:
                    vis_res = cv2.getWindowProperty('result', cv2.WND_PROP_VISIBLE)
                    vis_cap = cv2.getWindowProperty('capture', cv2.WND_PROP_VISIBLE)
                except Exception:
                    break
                if vis_res < 1 and vis_cap < 1:
                    break
                k = cv2.waitKey(100) & 0xFF
                if k == ord('q') or k == 27:
                    break

    except Exception as e:
        if args.json:
            print("AI_RESULT_JSON=" + json.dumps({"ok": False, "error": str(e)}))
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    finally:
        if cap:
            cap.release()
        if picam2:
            try:
                picam2.stop()
                try:
                    picam2.close()
                except Exception:
                    pass
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()