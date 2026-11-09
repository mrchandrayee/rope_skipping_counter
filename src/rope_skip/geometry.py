"""Floor-plane geometry and perspective overlay helpers.

The visible recording contains helpers for floor grid lines, a vanishing point,
a floor-height reference stick, world-plane quads and dashed projected segments.
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np

Point2 = tuple[float, float]
LineSeg = tuple[Point2, Point2]

def parse_points(value: object, *, name: str = "points") -> list[Point2]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of [x,y] points")
    out=[]
    for item in value:
        if not isinstance(item,(list,tuple)) or len(item)!=2:
            raise ValueError(f"{name} entries must be [x,y]")
        out.append((float(item[0]),float(item[1])))
    return out

def parse_line(value: object, *, name: str = "line") -> LineSeg:
    pts=parse_points(value,name=name)
    if len(pts)!=2:
        raise ValueError(f"{name} must contain exactly two points")
    return (pts[0],pts[1])

def parse_floor_lines(value: object) -> list[LineSeg]:
    """Parse floor grid lines; each entry is a two-point segment."""
    if value is None: return []
    return [parse_line(item,name="floor grid line") for item in value]

def parse_floor_height_ref(value: object):
    """Parse vertical stick reference: base/top image points and known height."""
    if value is None: return None
    if not isinstance(value,dict):
        raise ValueError("floor_height_ref must be a mapping")
    base=tuple(parse_points([value.get("base")],name="floor height ref base")[0])
    top=tuple(parse_points([value.get("top")],name="floor height ref top")[0])
    height=value.get("height_m",value.get("height"))
    if height is None: raise ValueError("floor_height_ref needs height_m")
    foot=value.get("foot_xz",(0,0))
    return base,top,float(height),(float(foot[0]),float(foot[1]))

def _m_or_mm(meters=None,millimeters=None):
    if meters is not None: return float(meters)
    if millimeters is not None: return float(millimeters)/1000.0
    return None

def _measured_quad_size(measured: dict):
    """Return (width along X, depth along Z) for measured quad."""
    width=_m_or_mm(measured.get("size_x_m"),measured.get("size_x_mm"))
    depth=_m_or_mm(measured.get("size_z_m"),measured.get("size_z_mm"))
    if width is None or depth is None:
        raise ValueError("measured quad requires size_x_m/size_x_mm and size_z_m/size_z_mm")
    return width,depth

def estimate_vanishing_point(lines: list[LineSeg]) -> Optional[Point2]:
    """Estimate a common vanishing point from pairwise line intersections."""
    intersections=[]
    for i,(a,b) in enumerate(lines):
        x1,y1=a; x2,y2=b
        for c,d in lines[i+1:]:
            x3,y3=c; x4,y4=d
            den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if abs(den)<1e-8: continue
            px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
            py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
            if -2000<=px<=4000 and -2000<=py<=4000:
                intersections.append((px,py))
    if not intersections: return None
    return (float(np.median([p[0] for p in intersections])),
            float(np.median([p[1] for p in intersections])))

def _point_side(line: LineSeg, point: Point2) -> float:
    """Signed side of point relative to line (2-D cross product)."""
    (x1,y1),(x2,y2)=line; x,y=point
    return (x2-x1)*(y-y1)-(y2-y1)*(x-x1)

def clip_segment_to_end_line(segment: LineSeg, end_line: LineSeg,
                             keep_point: Point2) -> Optional[LineSeg]:
    """Clip a floor ray at end_line, retaining the side containing keep_point."""
    a,b=segment; sa=_point_side(end_line,a); sb=_point_side(end_line,b)
    sk=_point_side(end_line,keep_point)
    if sa*sk<0: a,b=b,a; sa,sb=sb,sa
    if sb*sk>=0: return (a,b)
    den=sa-sb
    if abs(den)<1e-9: return (a,b)
    t=sa/den
    hit=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
    return (a,hit)

def draw_dashed_segment(image, segment: LineSeg, color=(0,255,0),
                        thickness=1, dash_px=14, gap_px=10):
    """Draw a dashed segment on an image."""
    (x1,y1),(x2,y2)=segment
    dx,dy=x2-x1,y2-y1
    length=float(np.hypot(dx,dy))
    if length<3: return
    ux,uy=dx/length,dy/length
    period=max(1,dash_px+gap_px)
    pos=0.0
    while pos<length:
        end=min(length,pos+dash_px)
        p1=(round(x1+ux*pos),round(y1+uy*pos))
        p2=(round(x1+ux*end),round(y1+uy*end))
        cv2.line(image,p1,p2,color,thickness,cv2.LINE_AA)
        pos+=period

def draw_floor_grid(frame, lines: list[LineSeg], vanishing_point=None,
                    end_line=None, keep_point=None, color=(0,220,255)):
    """Draw floor tile rays as dashed segments, clipped to end line."""
    if not lines: return frame
    overlay=frame.copy()
    h,w=frame.shape[:2]
    vp=vanishing_point
    for start,end in lines:
        seg=(start,end)
        if vp is not None:
            d1=(start[0]-vp[0])**2+(start[1]-vp[1])**2
            d2=(end[0]-vp[0])**2+(end[1]-vp[1])**2
            far=end if d2>d1 else start
            seg=(far,vp)
        if end_line is not None and keep_point is not None:
            seg=clip_segment_to_end_line(seg,end_line,keep_point)
            if seg is None: continue
        draw_dashed_segment(overlay,seg,color)
    return cv2.addWeighted(overlay,.65,frame,.35,0)

def standing_hip_place_reference(floor_quad, hip_xy, plane_points):
    """Draw a standing hip-to-floor reference and corresponding floor quad."""
    overlay=floor_quad.copy()
    if plane_points and len(plane_points)==4:
        pts=np.asarray(plane_points,dtype=np.int32)
        cv2.polylines(overlay,[pts],True,(0,255,180),2,cv2.LINE_AA)
    if hip_xy is not None:
        hx,hy=map(int,hip_xy)
        cv2.drawMarker(overlay,(hx,hy),(255,255,0),cv2.MARKER_CROSS,14,2)
    return overlay
