"""Common figure style for the PBGIA manuscript. Change FULL/HALF for a different journal class."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FULL=6.3     # in, article A4 text width (REVTeX two-column: 7.0)
HALF=3.1     # in
FS=8
C=dict(pbgia='#2a78d6',conv='#eb6834',smooth='#1baf7a',init='#eda100',prev='#8c8c86',ink='#2f2f2d',grid='#e4e3df',theory='#2f2f2d')
MK=dict(pbgia='o',conv='^',smooth='s',init='D',prev='d')
SFIELD=dict(cmap='coolwarm',vmin=-1,vmax=1)   # scalar component s, same in every figure
plt.rcParams.update({'font.size':FS,'axes.titlesize':FS,'axes.labelsize':FS,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42,
 'axes.edgecolor':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':C['grid'],'grid.linewidth':0.5,'lines.linewidth':1.2,'lines.markersize':4,
 'legend.frameon':False,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.02,'figure.dpi':150})
def label(ax,s,dx=-0.02,dy=1.02):
    """Panel label '(a)' at the upper-left corner outside the axes, same everywhere."""
    ax.text(dx,dy,f'({s})',transform=ax.transAxes,fontsize=FS,fontweight='bold',ha='right',va='bottom',color=C['ink'])
def imlabel(ax,s):
    """Panel label for image panels (no axes): upper-left inside a white box."""
    ax.text(0.02,0.98,f'({s})',transform=ax.transAxes,fontsize=FS,fontweight='bold',ha='left',va='top',color=C['ink'],
            bbox=dict(boxstyle='square,pad=0.15',fc='white',ec='none',alpha=0.85))
def noaxis(ax):
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
def save(fig,name,outdir):
    fig.savefig(f'{outdir}/{name}.pdf'); fig.savefig(f'{outdir}/{name}.png',dpi=200); plt.close(fig)
def legend_top(ax,ncol=2,**kw):
    """Legend in a row above the axes (keeps data unobstructed); panel label sits to its left."""
    return ax.legend(loc='lower left',bbox_to_anchor=(0.0,1.01),ncol=ncol,borderaxespad=0,handletextpad=0.3,columnspacing=0.9,**kw)
def label_top(ax,s):
    ax.text(-0.02,1.01,f'({s})',transform=ax.transAxes,fontsize=FS,fontweight='bold',ha='right',va='bottom',color=C['ink'])
