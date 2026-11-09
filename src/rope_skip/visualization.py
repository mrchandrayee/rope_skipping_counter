import cv2
def draw_overlay_text(frame,text,xy=(12,30),color=(255,255,255),scale=.6):
    cv2.putText(frame,str(text),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)
    return frame
def draw_text(frame,text,xy,color=(255,255,255),scale=.55):
    cv2.putText(frame,str(text),xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    return frame
def draw_hud(frame,count,fps,title="Rope Skipping",x=12,y=12,width=300,show_fps=True):
    h,w=frame.shape[:2]; width=max(100,min(width,w-x))
    overlay=frame.copy()
    cv2.rectangle(overlay,(x,y),(x+width,y+100),(12,24,42),-1)
    cv2.addWeighted(overlay,.78,frame,.22,0,frame)
    draw_overlay_text(frame,title,(x+12,y+26),(220,150,255),.65)
    draw_overlay_text(frame,f"JUMPS  {count}",(x+12,y+60),(0,255,255),.8)
    if show_fps: draw_text(frame,f"FPS {fps:.1f}",(x+12,y+84),(210,220,230),.45)
    return frame
