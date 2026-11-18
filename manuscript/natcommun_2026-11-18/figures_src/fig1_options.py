"""Four design options for Fig. 1: an overview schematic of the idea plus the concrete panels."""
import sys
sys.path.insert(0,'.')
from fig1_parts import *
from readout import boundary
W=7.1
# exact perimeter of the scene: trefoil-modulated outer boundary + hole + disk
th=np.linspace(0,2*np.pi,200001); r=38*(1+0.13*np.cos(3*th+0.4)); dr=-38*0.13*3*np.sin(3*th+0.4)
P_EX=np.trapezoid(np.sqrt(r*r+dr*dr),th)+2*np.pi*11+2*np.pi*13; P_HAT=boundary(S)

def bottom_strip(fig,y0,h,letters='bcde',x0=0.02,x1=0.98):
    gap=0.012; w=(x1-x0-3*gap)/4; axs=[fig.add_axes([x0+k*(w+gap),y0,w,h]) for k in range(4)]
    icon_binary(axs[0]); axs[0].set_title('source $f=\\pm1$',fontsize=7,pad=2)
    icon_s(axs[1]); axs[1].set_title('generated $s$',fontsize=7,pad=2)
    r0,c0,wd=wall_zoom(axs[2]); axs[1].add_patch(plt.Rectangle((c0-.5,r0-.5),wd,wd,fill=False,ec='black',lw=0.8)); axs[2].set_title('wall: $\\mathbf{v}$ along the inward normal',fontsize=7,pad=2)
    icon_tv(axs[3]); axs[3].set_title('readout $\\frac{1}{2}|\\nabla s|/m$',fontsize=7,pad=2)
    axs[3].text(0.04,0.05,f'$\\hat P$ = {P_HAT:.1f}\nexact {P_EX:.1f}',transform=axs[3].transAxes,color='white',fontsize=6.5,va='bottom')
    for a,l in zip(axs,letters): a.text(0.03,0.97,l,transform=a.transAxes,fontsize=9,fontweight='bold',va='top',color='white' if l in 'be' else 'black')
    return axs
def stage_title(ax,x,y,num,text,trans=None):
    ax.text(x,y,f'{num}',transform=trans or ax.transAxes,fontsize=7,fontweight='bold',color='white',ha='center',va='center',
            bbox=dict(boxstyle='circle,pad=0.25',fc=NAVY,ec='none'))
    ax.text(x+0.018,y,text,transform=trans or ax.transAxes,fontsize=7,fontweight='bold',color=NAVY,ha='left',va='center')

# =============================================================== A: pipeline
def option_A():
    fig=plt.figure(figsize=(W,5.0)); ov=overlay(fig)
    box(ov,0.01,0.43,0.98,0.555,fc='#f6f8fc',ec='#d5dbe8',r=0.015)
    ov.text(0.02,0.975,'a',fontsize=9,fontweight='bold',va='top')
    # stage 1: image as field
    a1=fig.add_axes([0.015,0.50,0.26,0.40]); plane(a1,np.where(MASK,0.55,-0.55),cmap=mpl.colormaps['RdBu_r'],alpha=0.9); field_arrows(a1,MASK,step=13,L=16); a1.set_aspect('equal'); a1.axis('off')
    stage_title(ov,0.045,0.935,'1','An image as a field')
    ov.text(0.035,0.475,'binary image $\\to$ Zeeman field $h f$,  $f=\\pm1$',fontsize=6.3,color='#33384a')
    arrow(ov,(0.255,0.70),(0.29,0.70),color=NAVY,lw=1.6,ms=11)
    # stage 2: rule
    stage_title(ov,0.31,0.935,'2','Alignment rule')
    box(ov,0.295,0.53,0.19,0.36,fc='white',ec='#c9cfdd',r=0.02,z=1)
    ov.text(0.39,0.80,'$\\mathbf{H}_v=J\\sum\\mathbf{v}_j+D\\,\\nabla s$',fontsize=7.2,ha='center',zorder=2)
    ov.text(0.39,0.735,'$H_s=J\\sum s_j+h\\,f$',fontsize=7.2,ha='center',zorder=2)
    ov.text(0.39,0.665,'$\\mathbf{n}\\leftarrow\\mathbf{H}/|\\mathbf{H}|$',fontsize=7.6,ha='center',color=ACC,zorder=2)
    ov.text(0.39,0.555,'no energy function:\nfixed points analysed directly',fontsize=6,ha='center',color='#5a6075',style='italic',zorder=2,linespacing=1.15)
    cg=fig.add_axes([0.335,0.585,0.11,0.06]); cg.set_xlim(-2.2,2.2); cg.set_ylim(-0.9,0.9); cg.axis('off')
    for xk,ang in ((-1.6,80),(-0.8,65),(0,0),(0.8,-65),(1.6,-80)):
        a_=np.deg2rad(ang); cg.annotate('',xy=(xk+0.35*np.sin(a_),0.55*np.cos(a_)),xytext=(xk-0.35*np.sin(a_),-0.55*np.cos(a_)),arrowprops=dict(arrowstyle='-|>,head_length=0.3,head_width=0.15',color=CM(0.5+0.5*np.cos(a_)),lw=1.2))
    cg.plot([-2,2],[-0.8,-0.8],color='#c9cfdd',lw=0.6)
    arrow(ov,(0.488,0.70),(0.52,0.70),color=NAVY,lw=1.6,ms=11)
    # stage 3: texture
    stage_title(ov,0.54,0.935,'3','Spin texture $\\mathbf{n}=(\\mathbf{v},s)$')
    a3=fig.add_axes([0.505,0.48,0.285,0.44]); plane(a3,S); spins3d(a3,VR,VC,S,L=12); a3.set_aspect('equal'); a3.axis('off')
    ov.text(0.53,0.475,'walls carry normals; $s$ follows the source',fontsize=6.3,color='#33384a')
    # stage 4: readouts
    stage_title(ov,0.80,0.935,'4','Readouts')
    ys=[0.79,0.655,0.52]; labels=[('Extent','$\\hat P=\\sum\\frac{1}{2}|\\nabla s|/m(\\varphi,\\kappa)$'),('Orientation','$k=\\frac{1}{4\\pi}\\oint d\\,\\arg q\\in\\frac{1}{2}\\mathbb{Z}$'),('Topology','$Q=\\chi(\\{s>0\\})$')]
    icons=[icon_tv,icon_director,icon_sphere]
    for y,(t_,eq),ic in zip(ys,labels,icons):
        box(ov,0.79,y-0.005,0.195,0.118,fc='white',ec='#c9cfdd',r=0.02,z=1)
        ia=fig.add_axes([0.795,y,0.06,0.108]); ic(ia)
        ov.text(0.862,y+0.08,t_,fontsize=6.8,fontweight='bold',color=NAVY,zorder=2); ov.text(0.862,y+0.03,eq,fontsize=6.2,zorder=2)
        arrow(ov,(0.775,0.70),(0.79,y+0.055),color='#8a93a8',lw=0.9,ms=7,rad=-0.15 if y>0.7 else (0.15 if y<0.6 else 0))
    bottom_strip(fig,0.03,0.33)
    return fig

# =============================================================== B: hub and spokes
def option_B():
    fig=plt.figure(figsize=(W,5.4)); ov=overlay(fig)
    box(ov,0.01,0.40,0.98,0.595,fc='#f6f8fc',ec='#d5dbe8',r=0.015); ov.text(0.02,0.98,'a',fontsize=9,fontweight='bold',va='top')
    c=fig.add_axes([0.27,0.48,0.46,0.44]); plane(c,S); spins3d(c,VR,VC,S,L=12); c.set_aspect('equal'); c.axis('off')
    ov.text(0.5,0.925,'One spin texture, generated from the image by a nonreciprocal alignment rule',fontsize=7.2,fontweight='bold',ha='center',color=NAVY)
    ov.text(0.5,0.455,'$\\mathbf{n}_i\\leftarrow\\mathbf{H}_i/|\\mathbf{H}_i|,\\quad \\mathbf{H}=(J\\sum\\mathbf{v}_j+D\\nabla s,\\; J\\sum s_j+hf)$',fontsize=7,ha='center')
    sats=[(0.025,0.74,'Wall','$w=\\ell F(\\kappa)$',icon_wall),(0.025,0.565,'Filter','$h>J(z_{\\rm out}-z_{\\rm in})$',icon_filter),(0.025,0.415,'Extent','$I_{\\rm TV}=\\frac{1}{2}\\int\\mathcal{A}(\\partial\\{s>t\\})dt$',icon_tv),
          (0.745,0.74,'Director','$k\\in\\frac{1}{2}\\mathbb{Z}$, conserved in $\\sigma$',icon_director),(0.745,0.565,'Scale','$\\sigma_c=0.441d$;  $n=\\frac{1}{4\\pi\\sigma^2}$',icon_scale),(0.745,0.415,'Degree','$Q=\\chi(\\{s>0\\})$ (Hopf)',icon_sphere)]
    for x,y,t_,eq,ic in sats:
        box(ov,x,y,0.23,0.15,fc='white',ec='#c9cfdd',r=0.02,z=1)
        ia=fig.add_axes([x+0.006,y+0.012,0.085,0.125]); ic(ia)
        ov.text(x+0.1,y+0.11,t_,fontsize=7,fontweight='bold',color=NAVY,zorder=2); ov.text(x+0.1,y+0.05,eq,fontsize=5.8,zorder=2)
        tx=x+0.23 if x<0.5 else x; arrow(ov,(0.5+(-0.15 if x<0.5 else 0.15),0.70),(tx,y+0.075),color='#8a93a8',lw=0.8,style='-',ms=1)
    bottom_strip(fig,0.03,0.30)
    return fig

# =============================================================== C: dictionary
def option_C():
    fig=plt.figure(figsize=(W,5.6)); ov=overlay(fig)
    box(ov,0.01,0.39,0.98,0.605,fc='#f6f8fc',ec='#d5dbe8',r=0.015); ov.text(0.02,0.985,'a',fontsize=9,fontweight='bold',va='top')
    ov.text(0.25,0.955,'Magnetization',fontsize=8,fontweight='bold',color=WARM,ha='center'); ov.text(0.75,0.955,'Image geometry',fontsize=8,fontweight='bold',color=ACC,ha='center')
    rows=[('Zeeman field $h f$','binary image $f=\\pm1$',icon_binary),
          ('domain wall, transverse moment $\\mathbf{v}$','boundary and its inward normal',icon_s),
          ('Ising stability $h>J(z_{\\rm out}-z_{\\rm in})$','scale-selective filtering',icon_filter),
          ('wall length (coarea of $s$)','perimeter / area without reference shape',icon_tv),
          ('nematic disclinations, $k\\in\\frac{1}{2}\\mathbb{Z}$','orientation defects across scales',icon_director),
          ('skyrmion number $Q$','Euler characteristic $\\chi$',icon_sphere)]
    y=0.885; dh=0.081
    for k,(lm,ri,ic) in enumerate(rows):
        yy=y-k*dh; fc='white' if k%2==0 else '#eef2fa'
        box(ov,0.03,yy-0.033,0.94,0.07,fc=fc,ec='none',r=0.01)
        ov.text(0.06,yy,lm,fontsize=6.8,va='center'); ov.text(0.62,yy,ri,fontsize=6.8,va='center')
        ia=fig.add_axes([0.465,yy-0.031,0.07,0.064]); ic(ia)
        arrow(ov,(0.40,yy),(0.46,yy),color=WARM,lw=1.0,ms=7); arrow(ov,(0.54,yy),(0.60,yy),color=ACC,lw=1.0,ms=7)
    ov.text(0.5,0.405,'nonreciprocal coupling $D$ widens walls ($\\kappa=D/\\sqrt{Jh}$) and has no energy function: every row is derived from fixed points',fontsize=6.3,ha='center',style='italic',color='#5a6075')
    bottom_strip(fig,0.03,0.29)
    return fig

# =============================================================== D: round trip
def option_D():
    fig=plt.figure(figsize=(W,5.2)); ov=overlay(fig)
    box(ov,0.01,0.41,0.98,0.585,fc='#f6f8fc',ec='#d5dbe8',r=0.015); ov.text(0.02,0.985,'a',fontsize=9,fontweight='bold',va='top')
    L=np.load('/home/claude/work/data/t51_ltem.npz'); lab=np.load('/home/claude/work/data/t53_labyrinth.npz')
    m1=fig.add_axes([0.03,0.70,0.135,0.22]); m1.imshow(L['A'][380:560,380:560],cmap='gray'); noaxis(m1); m1.set_title('skyrmions (LTEM)',fontsize=6.3,pad=1.5)
    m2=fig.add_axes([0.03,0.445,0.135,0.22]); m2.imshow(lab['raw'][40:180,40:180],cmap='gray'); noaxis(m2); m2.set_title('labyrinth (Kerr)',fontsize=6.3,pad=1.5)
    ov.text(0.097,0.945,'Magnet',fontsize=7.5,fontweight='bold',color=WARM,ha='center')
    arrow(ov,(0.175,0.68),(0.215,0.68),color=NAVY,lw=1.6,ms=11)
    a1=fig.add_axes([0.205,0.50,0.25,0.38]); plane(a1,np.where(MASK,0.55,-0.55),cmap=mpl.colormaps['RdBu_r'],alpha=0.9); field_arrows(a1,MASK,step=13,L=16); a1.set_aspect('equal'); a1.axis('off')
    ov.text(0.33,0.945,'Image read as a field',fontsize=7.5,fontweight='bold',color=NAVY,ha='center')
    arrow(ov,(0.455,0.68),(0.49,0.68),color=NAVY,lw=1.6,ms=11); ov.text(0.472,0.715,'$\\mathbf{n}\\leftarrow\\frac{\\mathbf{H}}{|\\mathbf{H}|}$',fontsize=7,ha='center')
    a3=fig.add_axes([0.47,0.48,0.29,0.44]); plane(a3,S); spins3d(a3,VR,VC,S,L=12); a3.set_aspect('equal'); a3.axis('off')
    ov.text(0.61,0.945,'Generated magnetization',fontsize=7.5,fontweight='bold',color=NAVY,ha='center')
    ov.text(0.875,0.945,'Geometry',fontsize=7.5,fontweight='bold',color=ACC,ha='center')
    for y,ic,t_ in ((0.78,icon_tv,'extent'),(0.625,icon_director,'orientation'),(0.47,icon_sphere,'topology')):
        ia=fig.add_axes([0.80,y,0.08,0.12]); ic(ia); ov.text(0.89,y+0.06,t_,fontsize=6.8,fontweight='bold',color=NAVY,va='center')
        arrow(ov,(0.745,0.68),(0.795,y+0.06),color='#8a93a8',lw=0.9,ms=7)
    arrow(ov,(0.86,0.455),(0.10,0.43),color=WARM,lw=1.3,ms=10,rad=-0.12)
    ov.text(0.48,0.418,'back to magnets: skyrmion number, disclinations, wall motion',fontsize=6.4,color=WARM,ha='center',style='italic',bbox=dict(fc='#f6f8fc',ec='none',pad=0.5))
    bottom_strip(fig,0.03,0.305)
    return fig

if __name__=='__main__':
    for nm,f in (('A',option_A),('B',option_B),('C',option_C),('D',option_D)):
        fig=f(); fig.savefig(f'../figures/Fig1_option{nm}.png',dpi=220); fig.savefig(f'../figures/Fig1_option{nm}.pdf'); plt.close(fig); print(nm)
