import PIL.Image
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision.ops.boxes import box_convert
from typing import Union


class Annotator:
    def __init__(self, im: Union[Image.Image, np.ndarray], line_width=None, font_size=None, font="Arial.ttf"):
        input_is_pil = isinstance(im, Image.Image)
        self.im = im if input_is_pil else Image.fromarray(im)
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        self.draw = ImageDraw.Draw(self.im)
        try:
            size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
            self.font = ImageFont.truetype(font, size)
        except Exception:
            self.font = ImageFont.load_default()
        self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        p1 = (box[0], box[1])
        self.draw.rectangle(box, width=self.lw, outline=color)  # box

        if label:
            w, h = self.font.getsize(label)  # text width, height
            outside = p1[1] - h >= 0  # label fits outside box
            self.draw.rectangle(
                (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                fill=color,
            )
            self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        if "\n" in text:
            lines = text.split("\n")
            _, h = self.font.getsize(text)
            for line in lines:
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im: Union[Image.Image, np.ndarray]):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        Image.fromarray(np.asarray(self.im)).show(title)

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, cv2.cvtColor(np.asarray(self.im),cv2.COLOR_RGB2BGR))



class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_images(images, targets, names) -> Annotator:
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # de-normalise (optional)
    if np.max(images[0]) <= 1:
        images *= 255

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y: y + h, x: x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, font='./utils/Arial.tff')
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = box_convert(torch.as_tensor(ti[:, 2:6]), in_fmt='cxcywh', out_fmt='xyxy').numpy().T
            classes = ti[:, 1].astype("int")
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[int(cls)] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{cls}" if labels else f"{cls} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)
    return annotator
