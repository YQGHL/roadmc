"""
RoadMC System Architecture Diagram Generator.
Saves to docs/architecture.png for READMEs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 1, figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

C = {"p1":"#3498DB","p2":"#E67E22","p3":"#27AE60","p4":"#8E44AD","p5":"#E74C3C","bg":"#F5F6FA","bo":"#2C3E50","ar":"#95A5A6"}

def bx(x,y,w,h,c,t,s="",fs=9,tc="w"):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",facecolor=c,edgecolor=C["bo"],lw=1,zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2+0.06,t,ha="center",va="center",fontsize=fs,fontweight="bold",color=tc,zorder=3)
    if s: ax.text(x+w/2,y+h/2-0.18,s,ha="center",va="center",fontsize=7,color=tc,alpha=0.8,zorder=3)

def ar(x1,y1,x2,y2,c=None):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color=c or C["ar"],lw=1.2,shrinkA=3,shrinkB=3),zorder=1)

def br(x,y,w,l,c):
    ax.plot([x,x,x+w,x+w],[y,y+0.12,y+0.12,y],color=c,lw=2.5,zorder=2)
    ax.text(x+w/2,y+0.25,l,ha="center",va="bottom",fontsize=10,fontweight="bold",color=c,zorder=2)

ax.add_patch(FancyBboxPatch((0.15,0.15),12.7,7.7,boxstyle="round,pad=0.04",facecolor=C["bg"],edgecolor=C["bo"],lw=1.2,zorder=0))
ax.text(6.5,7.7,"RoadMC System Architecture",ha="center",fontsize=15,fontweight="bold",color=C["bo"])
ax.text(6.5,7.45,"Physics-Simulation Driven Point Cloud Disease Detection",ha="center",fontsize=8,color=C["ar"])

br(0.4,7.2,5.0,"Phase 1: Data Generation",C["p1"])
br(5.7,7.2,3.2,"Phase 2: Core Network",C["p2"])
br(9.2,7.2,3.4,"Phases 3-5",C["p3"])

bx(0.6,5.4,1.5,0.55,C["p1"],"config.py","Dataclass config",fs=8)
bx(0.6,4.3,1.5,0.55,C["p1"],"primitives.py","13 physics functions",fs=8)
bx(0.6,3.2,1.5,0.55,C["p1"],"generator.py","SyntheticRoadDataset",fs=8)
ar(1.35,5.4,1.35,4.85); ar(1.35,4.3,1.35,3.75)

for i,f in enumerate(["ISO 8608 PSD surface","fBm micro-texture","11 disease models","LiDAR noise + voxel"]):
    ax.plot(2.5,3.1-i*0.35,'o',color=C["p1"],ms=4,zorder=2)
    ax.text(2.8,3.1-i*0.35,f,fontsize=6.5,color=C["bo"],va="center",zorder=2)
ar(2.1,3.5,5.7,4.8)

bx(5.9,5.6,2.8,0.55,C["p2"],"Swin3D Backbone","C=96 -> 192 -> 384 -> 768",fs=9)
for i,(n,xp) in enumerate([("S0",5.9),("S1",6.6),("S2",7.3),("S3",8.0)]):
    bx(xp,4.7,0.6,0.45,"#F39C12",n,"d=2" if i<2 else "d=6" if i==2 else "d=2",fs=6)
    if i<3: ar(6.5+i*0.7,4.92,6.5+i*0.7+0.65,4.92)
ar(7.3,5.6,7.3,5.15); ar(7.3,4.7,7.3,4.5)

bx(5.9,3.8,2.8,0.7,C["p2"],"Transformer Block","LN->WindowAttn->+x->LN->FFN->+x->MHC",fs=8)
bx(5.9,2.8,2.8,0.55,"#D35400","Segmentation Head","FCN + Skip -> (B,N,38)",fs=8)
ar(7.3,3.8,7.3,3.35); ar(7.3,2.8,7.3,2.2)
bx(9.5,2.8,1.8,0.4,"#C0392B","MHCConnection","Sinkhorn-Knopp",fs=7)
ar(8.7,3.0,9.5,3.0)

bx(9.4,4.7,2.9,0.55,C["p3"],"StyleTransferGen","DGCNN EdgeConv encoder",fs=8)
bx(9.4,3.9,2.9,0.5,C["p3"],"WGANDiscriminator","PointNet WGAN-GP critic",fs=8)
ar(10.85,4.7,10.85,4.45)

bx(9.4,1.3,2.9,0.45,C["p4"],"RoadMCDataModule","Lightning DataModule",fs=8)
bx(9.4,0.65,2.9,0.4,C["p4"],"RealRoadDataset",".ply/.npy -> JTG labels",fs=7)

bx(0.6,0.65,3.8,1.4,C["p5"],"Training",fs=9)
for i,(m,d) in enumerate([("baseline","synthetic only"),("gan_enhanced","GAN then mixed"),("end2end","alternating GAN+seg")]):
    ax.text(0.9,1.55-i*0.3,f"  {m}: {d}",fontsize=6.5,color="w",zorder=3)
bx(0.6,0.1,3.8,0.35,"#C0392B","evaluate.py: JTG per-class report (IoU/recall/precision)",fs=6.5)

ar(4.4,1.5,9.4,1.5,"#8E44AD")
ar(8.7,4.15,9.4,4.15)

for i,(c,l) in enumerate(zip([C["p1"],C["p2"],C["p3"],C["p4"],C["p5"]],
    ["P1: Data","P2: Network","P3: GAN","P4: DataLoad","P5: Train/Eval"])):
    ax.plot(2.0+i*1.8,0.05,'s',color=c,ms=7,zorder=2)
    ax.text(2.0+i*1.8,-0.08,l,fontsize=5.5,ha="center",color=C["bo"],zorder=2)

plt.savefig("docs/architecture.png",dpi=200,bbox_inches="tight",facecolor="white")
print("[PASS] docs/architecture.png")
