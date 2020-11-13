import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def pybob_plot_aspect_slope_fit(xdata,ydata2,mysamp,xp,yp,sp,xadj,yadj,zadj,title,pp):

    fig = plt.figure(figsize=(7, 5), dpi=300)
    # fig.suptitle(title, fontsize=14)
    plt.title(title, fontsize=14)
    plt.plot(xdata[mysamp], ydata2[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
    plt.plot(xp, np.zeros(xp.size), 'k', ms=3)
    # plt.plot(xp, np.divide(yp,sp), 'r-', ms=2)
    plt.plot(xp, np.divide(yp, sp), 'r-', ms=2)

    plt.xlim(0, 360)
    ymin, ymax = plt.ylim((np.nanmean(ydata2[mysamp])) - 2 * np.nanstd(ydata2[mysamp]),
                          (np.nanmean(ydata2[mysamp])) + 2 * np.nanstd(ydata2[mysamp]))

    # plt.axis([0, 360, -200, 200])
    plt.xlabel('Aspect [degrees]')
    plt.ylabel('dH / tan(slope)')
    numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
    plt.text(0.05, 0.15, '$\Delta$x: ' + ('{:.1f} m'.format(xadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.1, '$\Delta$y: ' + ('{:.1f} m'.format(yadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.05, '$\Delta$z: ' + ('{:.1f} m'.format(zadj)).rjust(numwidth),
             fontsize=12, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    pp.savefig(fig, dpi=200)

def pybob_false_hillshade(dH, title, pp, clim=(-20, 20)):
    niceext = np.array([dH.xmin, dH.xmax, dH.ymin, dH.ymax]) / 1000.
    mykeep = np.logical_and.reduce((np.isfinite(dH.img), (np.abs(dH.img) < np.nanstd(dH.img) * 3)))
    dH_vec = dH.img[mykeep]

    fig = plt.figure(figsize=(7, 5), dpi=300)
    ax = plt.gca()

    im1 = ax.imshow(dH.img, extent=niceext)
    im1.set_clim(clim[0], clim[1])
    im1.set_cmap('Greys')
    #    if np.sum(np.isfinite(dH_vec))<10:
    #        print("Error for statistics in false_hillshade")
    #    else:
    plt.title(title, fontsize=14)

    numwid = max([len('{:.1f} m'.format(np.mean(dH_vec))),
                  len('{:.1f} m'.format(np.median(dH_vec))), len('{:.1f} m'.format(np.std(dH_vec)))])
    plt.annotate('MEAN:'.ljust(8) + ('{:.1f} m'.format(np.mean(dH_vec))).rjust(numwid), xy=(0.65, 0.95),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')
    plt.annotate('MEDIAN:'.ljust(8) + ('{:.1f} m'.format(np.median(dH_vec))).rjust(numwid),
                 xy=(0.65, 0.90), xycoords='axes fraction', fontsize=12, fontweight='bold',
                 color='red', family='monospace')
    plt.annotate('STD:'.ljust(8) + ('{:.1f} m'.format(np.std(dH_vec))).rjust(numwid), xy=(0.65, 0.85),
                 xycoords='axes fraction', fontsize=12, fontweight='bold', color='red', family='monospace')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    # plt.colorbar(im1)

    plt.tight_layout()
    pp.savefig(fig, dpi=300)