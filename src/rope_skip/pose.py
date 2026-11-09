from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO

@dataclass
class PoseResult:
    box: Optional[tuple[float,float,float,float]]
    keypoints: np.ndarray
    keypoint_conf: np.ndarray
    confidence: float

class PoseEstimator:
    """YOLO pose wrapper returning COCO keypoints and person boxes."""
    def __init__(self, model="yolo11n-pose.pt", device=None, imgsz=960, conf=.25):
        self.model=YOLO(model); self.device=device; self.imgsz=imgsz; self.conf=conf
    def predict(self, frame):
        result=self.model.predict(frame,device=self.device,imgsz=self.imgsz,
                                  conf=self.conf,verbose=False)[0]
        poses=[]
        if result.keypoints is None or result.boxes is None: return poses
        xy=result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is not None:
            kc=result.keypoints.conf.cpu().numpy()
        else:
            kc=np.ones(xy.shape[:2],dtype=float)
        boxes=result.boxes.xyxy.cpu().numpy()
        confs=result.boxes.conf.cpu().numpy()
        for i in range(min(len(xy),len(boxes))):
            poses.append(PoseResult(tuple(map(float,boxes[i])),xy[i],kc[i],float(confs[i])))
        return poses

def midpoint_from_pose(pose: PoseResult, a: int, b: int, min_conf=.3):
    if pose.keypoint_conf[a]<min_conf or pose.keypoint_conf[b]<min_conf: return None
    return tuple(((pose.keypoints[a]+pose.keypoints[b])/2).tolist())

def pose_anchor_xy(pose: PoseResult, *, min_keypoint_conf=.3, mode="ankle"):
    """Prefer ankle midpoint, else hip midpoint, else person-box bottom center."""
    if mode=="ankle":
        point=midpoint_from_pose(pose,15,16,min_conf=min_keypoint_conf)
        if point is not None: return point
    hip=midpoint_from_pose(pose,11,12,min_conf=min_keypoint_conf)
    if hip is not None and mode!="box": return hip
    if pose.box is not None:
        x1,y1,x2,y2=pose.box
        return ((x1+x2)/2,y2)
    return None

def draw_pose(frame, pose: PoseResult, min_conf=.3):
    # COCO skeleton edges
    edges=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
           (11,12),(11,13),(13,15),(12,14),(14,16)]
    pts=pose.keypoints
    for a,b in edges:
        if pose.keypoint_conf[a]>=min_conf and pose.keypoint_conf[b]>=min_conf:
            cv2.line(frame,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),
                     (60,220,80),2,cv2.LINE_AA)
    for i,p in enumerate(pts):
        if pose.keypoint_conf[i]>=min_conf:
            cv2.circle(frame,tuple(p.astype(int)),3,(0,220,255),-1,cv2.LINE_AA)
    return frame
