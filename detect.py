import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

from nvidia.dali.plugin.pytorch import DALIGenericIterator


# Silence PyTorch warnings 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

cudnn.benchmark = True  # set True to speed up constant image size inference
device = torch.device('cuda:0')

def setup_model(checkpoint_path='data/models/yolov7.pt'):
    # Load model
    model = attempt_load(checkpoint_path, map_location=device)  # load FP32 model
    model = model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    return model, names, colors


def setup_output(output_path, fps = 24.0, w = 1280, h = 720, path='output.mp4'):
    vid_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (w, h))
    
    return vid_writer


def draw_predictions(names, colors, img, im0s, det):
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)


def detect_native(dataset, output_path):
    model, names, colors = setup_model()
    output = setup_output(output_path)

    
    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        # img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred)

        # Process detections
        for _, det in enumerate(pred):  # detections per image
            draw_predictions(names, colors, img, im0s, det)

            output.write(im0s)

    print(f'Done')


def detect_dali(dataset, output_path):
    model, names, colors = setup_model()
    output = setup_output(output_path)
    
    for i, data in enumerate(dataset):
        batch = data[0]['frames'][0]
        raw = data[0]['raw'][0]

        for img, im0s in zip(batch, raw):
            img = torch.movedim(img, 2, 0)
            # img = torch.from_numpy(img).to(device)
            img = img.half()
            # img = img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred)

            im0s = im0s.cpu().numpy()

            # Process detections
            for _, det in enumerate(pred):  # detections per image
                draw_predictions(names, colors, img, im0s, det)

                output.write(im0s)

    print(f'Done')


def detect(dataset, output_path):
    if isinstance(dataset, DALIGenericIterator):
        detect_dali(dataset, output_path)
    else:
        detect_native(dataset, output_path)
