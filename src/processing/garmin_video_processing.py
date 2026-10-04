#!/usr/bin/env python3
import argparse
import cv2
import pytesseract
import easyocr
import re
import json
import numpy as np
import os
from pathlib import Path
import sys
import uuid
from typing import Dict, List

import os

def clear_and_print(text):
    # 'nt' is for Windows (uses 'cls'), others use 'clear'
    os.system('cls' if os.name == 'nt' else 'clear')
    print(text)

#
#   This generates pixel coordinates of the red marker within the g-force graph with respect to time (video frame number)
#   Supply your Garmin Catalyst video files. To ensure compatibility, your video files must not be post-processed in
#   a way that modifies the nature of the video data, such as changing resolution, framerate, color, etc. 
#   Trimming the original source video to a shorter video is OK.
#   
#   Comments:
#       I tried to map the red dots position with the G force magnitude based on pixel location on the video overlay
#       but the distance between the different rings of the g-force graph are different between each ring so it's not a simple linear 
#       deduction of pixel coordinate == gforce value. 
#
#       The 'unit' of the x/y values are in are pixels related to the template image, not the g-force magnitude as recorded by Catalyst. 
#       As a result, these pixel values, which does correspond to the graphs that the Catalyst generates but lack units,
#       is normalized between [-1, 1]. This has no affect on the shape of the graph, but makes viewing the the units on 
#       the x/y axis less strange: imagine values like 151x58, verses some value between -1, 1. 
#       Interpreting the results is more of matter of observing the shape of the curve rather than knowing what 
#       specific G-force magnitude was felt at a giving point in time. 

#       The Garmin Catalyst device/app will tell you the actual value if you really want to know. Though it would be nice to 
#       implement support for the pixel -> g-force conversion.
#
#   Future work:
#       I wanted to read the text present in the video overlay, like speed, time, delta, so I tried using the 
#       OCR library, Tesseract. However it was very inconsistent producing correct output. The font used by Garmin
#       does not seem compatible with Tesseract. An implementation I have in mind would be to use the various .pngs under
#       processing/templates and use opencv to match on shapes within the regions where times are visibile to
#       deduce what the text says.
#

class FrameData:
    def __init__(self, x, y, frame):
        self.x = x
        self.y = y
        self.frame = frame

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.frame
        }

def process_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--data_source', type=str, required=True)
    parser.add_argument('-p', '--data_file_path', type=str, required=True)
    parser.add_argument('-t', '--template_path', type=str, help="Supply the path to where the template image is located", required=True)
    parser.add_argument('-o', '--output_path', type=str, help="Supply the path of the OS's, app specific, folder so as to save application generated data locally to the system", required=True)
    args = parser.parse_args()

    sanity_check(args)

    return args


def sanity_check(args):
    if args.data_source != 'garmincatalyst':
        sys.stderr.write(f"Unexpected data_source {args.data_source}. Currently, only `garmincatalyst` is allowed.\n")
        exit(-1)

    if not os.path.exists(args.data_file_path):
        sys.stderr.write(f"The video file specified, {args.data_file_path}, does not exist.\n")
        exit(-1)
    if not args.data_file_path.endswith('.mp4'):
        sys.stderr.write(f"Video file, {args.data_file_path} should be a .mp4 file.\n")
        exit(-1)
    if os.path.getsize(args.data_file_path) == 0:
        sys.stderr.write(f"Video file, {args.data_file_path} is empty.\n")
        exit(-1)

    if not os.path.exists(args.template_path):
        sys.stderr.write(f"The template image file, {args.template_path}, does not exist.\n")
        exit(-1)
    if not args.template_path.endswith('.png'):
        sys.stderr.write(f"Template image file, {args.template_path}, should be a .png.\n")
        exit(-1)
    if os.path.getsize(args.template_path) == 0:
        sys.stderr.write(f"Template image file, {args.template_path}, is empty.\n")
        exit(-1)


def get_video_frame_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (num_frames, fps)


def process_video(video_path: str, template_path: str) -> List[FrameData]:
    def extract_text(reader, roi, data_type, allowlist='0123456789:.', debug=False):
        results = reader.readtext(roi, allowlist=allowlist)
        retval = "-1"
        for (bbox, text, prob) in results:
            if prob > 0.2 and len(results) == 1:
                retval = text

        if retval == "-1":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresholded = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            custom_config = f'--psm 6 -c tessedit_char_whitelist={allowlist}'
            retval = pytesseract.image_to_string(thresholded, config=custom_config).strip()

        return retval
    def extract_gforce_data(gforce_roi, gforce_mask, frame_data):
        gforce_circular_roi = cv2.bitwise_and(gforce_roi, gforce_roi, mask=gforce_mask)
        
        gforce_hsv = cv2.cvtColor(gforce_circular_roi, cv2.COLOR_BGR2HSV)
    
        mask1 = cv2.inRange(gforce_hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(gforce_hsv, lower_red_2, upper_red_2)
        red_mask = mask1 + mask2

        moments = cv2.moments(red_mask)
        if moments['m00'] != 0:
            # FIXME: cX and cY are with respect to the roi's pixel dimension.
            cX = -int(moments['m10'] / moments['m00'])
            cY = int(moments['m01'] / moments['m00'])
            frame_data.append(FrameData(cX, cY, frame_number))
        return cX, cY

    frame_data: List[FrameData] = []
    
    gforce_mask_results = generate_gforce_mask(video_path, template_path)
    gforce_roi = gforce_mask_results['roi']
    gforce_mask = gforce_mask_results['mask']

    gforce_roi_height, gforce_roi_width, _ = gforce_roi.shape

    # HSV match red color - G-force meter shows a red dot moving based on gforces
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    frame_number = 0
    cap = cv2.VideoCapture(video_path)
    start_lap = False
    minute_memory = 0
    segment_memory = 0
    reader = easyocr.Reader(['en'])

    with open(csv_path, "w") as f:
        f.write("frame_number;elapsed_time;segment;speed;delta;gforce_cX;gforce_cY\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        clear_and_print(f"processing frame {frame_number}")
        frame_height, frame_width, _ = frame.shape

        OFFSET = 25
        start_y = frame_height - OFFSET - gforce_roi_height
        end_y = frame_height - OFFSET
        start_x = frame_width - OFFSET - gforce_roi_width
        end_x = frame_width - OFFSET

        gforce_roi = frame[start_y:end_y, start_x:end_x]
        speed_roi = frame[880:935, 90:225]
        timestamp_roi = frame[1000:end_y, 305:600]
        segment_roi = frame[0:80, 1295:1360]
        delta_roi = frame[1000:end_y, 1400:1520]

        speed_text = extract_text(reader, speed_roi, "speed")
        timestamp_text = extract_text(reader, timestamp_roi, "timestamp")

        if timestamp_text != '' and timestamp_text != "-1":
            components = re.split(r'[-:.]', timestamp_text)
            if len(components) != 3:
                timestamp_text = "-1"
                continue
            minute, sec, msec = components
            if int(minute) == 0 and minute_memory > 0 and start_lap:
                break
            elif int(minute) == 0 and minute_memory > 0:
                start_lap = True
            minute_memory = int(minute)

        if start_lap:
            gforce_cX, gforce_cY = extract_gforce_data(gforce_roi, gforce_mask, frame_data)
            segment_text = extract_text(reader, segment_roi, "segment", allowlist='0123456789', debug=True)
            if segment_text != "-1" and int(segment_text) - segment_memory <= -9:
                segment_roi = segment_roi = frame[0:80, 1275:1360]
                segment_text = extract_text(reader, segment_roi, "segment", allowlist='0123456789', debug=True)
                #segment_memory = int(segment_text)
            elif segment_text == "-1":
                segment_text = f"{segment_memory}"
            segment_memory = int(segment_text)
            delta_text = extract_text(reader, delta_roi, "delta", allowlist='0123456789+-.')
            
            with open(csv_path, "+a") as f:
                f.write(f"{frame_number};{timestamp_text};{segment_text};{speed_text};{delta_text};{gforce_cX};{gforce_cY}\n")
            
    cap.release()

    return frame_data 

def sanitize_csv_file(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove any empty lines
    for i, line in enumerate(lines):
        prevLine = lines[i-1] if i > 1 else ""
        currLine = line
        nextLine = lines[i+1] if i < len(lines) - 1 else ""

        if prevLine != "" and currLine != "" and nextLine != "":
            box = [prevLine.strip().split(';'), currLine.strip().split(';'), nextLine.strip().split(';')]
            # for each row, look at the ith index and see if the currentLine is "-1".
            # if it is, grab the value from the ith index in the prevLine 
            for j in range(len(box[0])):
                if box[1][j] == "-1":
                    box[1][j] = box[0][j]
            lines[i] = ";".join(box[1])

        lines[i] = f"{lines[i].strip()}\n"

    with open(file_path, 'w') as f:
        f.writelines(lines)

def generate_gforce_mask(video_path: str, template_path: str) -> Dict[str, cv2.Mat]:
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    roi_height, roi_width, _ = template.shape
    mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    center = (roi_height // 2, roi_height // 2)
    radius = min(roi_width, roi_height) // 2
    cv2.circle(mask, center, radius, color=255, thickness=-1)

    # Garmin Catalyst video files all have the same static overlay so any frame 
    # from any point in the video will have the same overlay
    cap = cv2.VideoCapture(video_path)
    _, sample_frame = cap.read()
    cap.release()

    sample_height, sample_width, _ = sample_frame.shape

    # offset number determined by visually inspecting the frame via cv2.imshow()
    OFFSET = 25
    start_y = sample_height - OFFSET - roi_height
    end_y = sample_height - OFFSET
    start_x = sample_width - OFFSET - roi_width
    end_x = sample_width - OFFSET

    roi_frame = sample_frame[start_y : end_y, start_x : end_x]
    roi = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)

    return {'roi': roi, 'mask': mask}


def jsonify_results(num_frames, fps, frame_data):
    json_results = []
    for frame in frame_data:
        json_results.append(frame.to_dict())

    processing_response = {
        'data': {
            'num_frames': num_frames,
            'fps': fps,
            'trace': json_results,
        },
    }

    return json.dumps(processing_response)

if __name__ == "__main__":
    args = process_cli_args()

    num_frames, fps = get_video_frame_info(args.data_file_path)
    csv_path = f"{args.output_path}/{Path(args.data_file_path).parent.name}_{Path(args.data_file_path).stem}.csv"
    frame_data = process_video(args.data_file_path, args.template_path)
    #json_output = jsonify_results(num_frames, fps, frame_data) 
    sanitize_csv_file(csv_path)

    #generated_uuid = str(uuid.uuid4())
    #file_path = f"{args.output_path}/video_{generated_uuid}.json"
    
    #with open(file_path, 'w') as f:
    #    f.writelines(json_output)

    #print(f"{file_path}")