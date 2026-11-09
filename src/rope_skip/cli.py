from __future__ import annotations
import argparse, time
from pathlib import Path
import cv2, yaml
from .pose import PoseEstimator, pose_anchor_xy, draw_pose
from .counting import JumpCounter
from .geometry import parse_floor_lines, parse_line, estimate_vanishing_point, draw_floor_grid
from .visualization import draw_hud, draw_overlay_text

def load_config(path):
    with open(path,encoding="utf-8") as f: return yaml.safe_load(f)

def main():
    ap=argparse.ArgumentParser(description="Rope skipping / jump counter using pose estimation")
    ap.add_argument("--config",default="configs/rope_skipping.yaml")
    ap.add_argument("--source",help="Video path, image path, folder, or camera index")
    ap.add_argument("--output",help="Output directory")
    ap.add_argument("--no-show",action="store_true")
    ap.add_argument("--no-save",action="store_true")
    args=ap.parse_args()
    cfg=load_config(args.config); run=cfg["run"]
    if args.source is not None: run["source"]=args.source
    if args.output: run["output"]=args.output
    if args.no_show: run["show"]=False
    if args.no_save: run["save_video"]=False
    out=Path(run.get("output","runs/rope_skipping")); out.mkdir(parents=True,exist_ok=True)
    pose_cfg=cfg["pose"]; rc=run
    estimator=PoseEstimator(pose_cfg["model"],rc.get("device"),rc.get("imgsz",960),rc.get("confidence",.25))
    counter=JumpCounter(pose_cfg.get("jump_threshold_px",22),
        pose_cfg.get("min_jump_interval_frames",5),pose_cfg.get("cooldown_frames",8),
        pose_cfg.get("smoothing",.65))
    floor=cfg.get("floor",{})
    lines=parse_floor_lines(floor.get("grid_lines"))
    vp=floor.get("vanishing_point") or estimate_vanishing_point(lines)
    end_line=parse_line(floor["end_line"]) if floor.get("end_line") else None
    keep=(0.0,0.0)
    source=run.get("source",0)
    src=Path(str(source))
    if src.is_dir():
        paths=[p for p in sorted(src.rglob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
        for path in paths:
            frame=cv2.imread(str(path))
            if frame is None: continue
            vis=process(frame,estimator,counter,pose_cfg,cfg,floor,lines,vp,end_line,keep,0)
            cv2.imwrite(str(out/f"{path.stem}_counted.jpg"),vis)
        print(f"Processed images: {len(paths)}")
        return
    if src.suffix.lower() in {".jpg",".jpeg",".png",".bmp"} and src.exists():
        frame=cv2.imread(str(src))
        vis=process(frame,estimator,counter,pose_cfg,cfg,floor,lines,vp,end_line,keep,0)
        cv2.imwrite(str(out/f"{src.stem}_counted.jpg"),vis)
        return
    camera=str(source).isdigit() and not src.exists()
    cap=cv2.VideoCapture(int(source) if camera else str(source))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input: {source}")
    fps_src=cap.get(cv2.CAP_PROP_FPS) or 25
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=None
    if run.get("save_video",True):
        writer=cv2.VideoWriter(str(out/"rope_skipping_output.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps_src,(w,h))
    started=time.time(); n=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            fps=(n+1)/max(time.time()-started,1e-6)
            vis=process(frame,estimator,counter,pose_cfg,cfg,floor,lines,vp,end_line,keep,fps)
            if writer: writer.write(vis)
            if run.get("show",True):
                cv2.imshow("Rope Skipping Counter",vis)
                if cv2.waitKey(1)&0xFF==ord("q"): break
            n+=1
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {n} frames. Total jumps: {counter.state.count}. Output: {out}")

def process(frame,estimator,counter,pose_cfg,cfg,floor,lines,vp,end_line,keep,fps):
    vis=frame.copy()
    poses=estimator.predict(frame)
    if poses:
        pose=max(poses,key=lambda p:p.confidence)
        draw_pose(vis,pose,pose_cfg.get("keypoint_confidence",.3))
        anchor=pose_anchor_xy(pose,min_keypoint_conf=pose_cfg.get("keypoint_confidence",.3),
                              mode=pose_cfg.get("count_mode","ankle"))
        count,jumped=counter.update(anchor)
        if anchor:
            cv2.circle(vis,tuple(map(int,anchor)),6,(0,255,255),-1)
    else:
        count=counter.state.count
    if floor.get("draw_grid",True):
        vis=draw_floor_grid(vis,lines,vp,end_line,keep,tuple(floor.get("grid_color",[0,220,255])))
    hud=cfg.get("hud",{})
    draw_hud(vis,count,fps,hud.get("title","Rope Skipping"),hud.get("x",12),
             hud.get("y",12),hud.get("width",300),hud.get("show_fps",True))
    return vis

if __name__=="__main__": main()
