import os
import rasterio as rio
import geopandas as gpd
import richdem as rd
import numpy as np
from scipy import optimize
from matplotlib.backends.backend_pdf import PdfPages

from DemUtils.spatial_tools import nmad, rmse
from DemUtils.plot_tools import pybob_false_hillshade, pybob_plot_aspect_slope_fit

from GeoUtils.vector_tools import Vector
from GeoUtils.raster_tools import Raster

def rio_to_rda(ds:rio.DatasetReader)->rd.rdarray:
    """
    Get georeferenced richDEM array from rasterio dataset
    :param ds: DEM
    :return: DEM
    """

    arr = ds.read(1)
    rda = rd.rdarray(arr, no_data=ds.get_nodatavals()[0])
    rda.geotransform = ds.get_transform()
    rda.projection = ds.get_gcps()

    return rda

def get_terrainattr(ds:rio.DatasetReader,attrib='slope_degrees')->rd.rdarray:
    """
    Derive terrain attribute for DEM opened with rasterio. One of "slope_degrees", "slope_percentage", "aspect",
    "profile_curvature", "planform_curvature", "curvature" and others (see richDEM documentation)
    :param ds: DEM
    :param attrib: terrain attribute
    :return:
    """

    rda = rio_to_rda(ds)
    terrattr = rd.TerrainAttribute(rda, attrib=attrib)

    return terrattr


def pybob_filter(stable_mask, slope, aspect, master, slave):

    stan = np.tan(np.radians(slope)).astype(np.float32)
    dH = master - slave
    dH[stable_mask] = np.nan
    mykeep = ((np.absolute(dH) < 200.0) & np.isfinite(dH) &
              (slope > 3.0) & (dH != 0.0) & (aspect >= 0))
    dH[np.invert(mykeep)] = np.nan
    xdata = aspect[mykeep]
    ydata = dH[mykeep]
    sdata = stan[mykeep]

    return dH, xdata, ydata, sdata

def pybob_nuth_fit(xdata, ydata, sdata, title, pp):

    xdata = xdata.astype(np.float32)
    ydata = ydata.astype(np.float32)
    sdata = sdata.astype(np.float32)

    ydata2 = np.divide(ydata, sdata)

    # fit using equation 3 of Nuth and Kääb, 2011
    def fitfun(p, x, s):
        return p[0] * np.cos(np.radians(p[1] - x)) * s + p[2]

    def errfun(p, x, s, y):
        return fitfun(p, x, s) - y

    if xdata.size > 20000:
        mysamp = np.random.randint(0, xdata.size, 20000,dtype=np.int64)
    else:
        mysamp = np.arange(0, xdata.size,dtype=np.int64)

    p0 = [1, 1, -1]
    myresults = optimize.least_squares(errfun, p0, args=(xdata[mysamp], sdata[mysamp], ydata[mysamp]), method='trf',
                                       loss='soft_l1', f_scale=0.1, ftol=1E-8, xtol=1E-8)
    p1 = myresults.x

    # convert to shift parameters in cartesian coordinates
    xadj = p1[0] * np.sin(np.radians(p1[1]))
    yadj = p1[0] * np.cos(np.radians(p1[1]))
    zadj = p1[2]  # * sdata.mean(axis=0)

    xp = np.linspace(0, 360, 361)
    sp = np.ones(xp.size) + np.nanmean(sdata[mysamp])
    p1[2] = np.divide(p1[2], np.nanmean(sdata[mysamp]))
    yp = fitfun(p1, xp, sp)

    pybob_plot_aspect_slope_fit(xdata,ydata2,mysamp,xp,yp,sp,xadj,yadj,zadj,title,pp)

    return xadj, yadj, zadj


def pybob_compute_shift_nuth(master_dem:rio.DatasetReader, slave_dem:rio.DatasetReader, exc_shp:gpd.GeoDataFrame, inc_shp=gpd.GeoDataFrame,outdir='.',magnlimit=2.):

    """
    :param master_dem: Master DEM
    :param slave_dem: Slave DEM
    :param exc_shp: Glacier shapefile to mask
    :param inc_shp: Land shapefile to keep
    :param outdir: Output directory
    :param magnlimit: Magnitude threshold for determining termination of co-registration algorithm, calculated as
        sum in quadrature of dx, dy, dz shifts. Default is 2 m.
    :return:
    """

    #reproject slave DEM to master
    # TODO: adapt this to GeoUtils
    this_slave = slave_dem.reproject(master_dem)

    #get slope and spect from master DEM
    slope = get_terrainattr(master_dem,attrib='slope_degrees')
    aspect = get_terrainattr(master_dem,attrib='aspect')

    #get mask of stable terrain from inclusive and exclusive shapefiles
    # TODO: adapt this to GeoUtils
    excl_vec = Vector(exc_shp)
    incl_vec = Vector(inc_shp)
    mask_glacier = excl_vec.create_mask(master_dem)
    mask_land = incl_vec.create_mask(master_dem)
    stable_mask = np.logical_and(~mask_glacier,mask_land)

    # # make a file to save the coregistration parameters and statistics to.
    paramf = open(outdir + os.path.sep + 'coreg_params.txt', 'w')
    statsf = open(outdir + os.path.sep + 'stats.txt', 'w')
    pp = PdfPages(outdir + os.path.sep + 'CoRegistration_Results.pdf')

    #initialize iterative parameters
    mythresh, mystd = (np.float32(200) for i in range(2))
    mycount = 0
    tot_dx, tot_dy, tot_dz = (np.float64(0) for i in range(3))
    magnthresh = 200

    #iterative fit
    while mythresh > 2 and magnthresh > magnlimit:
        mycount += 1
        print("Running iteration #{}".format(mycount))
        print("Running iteration #{}".format(mycount), file=paramf)

        #filter data
        dH, xdata, ydata, sdata = pybob_filter(stable_mask, slope, aspect, master_dem, this_slave)

        #criteria to stop if not enough data
        if np.logical_or.reduce((np.sum(np.isfinite(xdata.flatten())) < 100,
                                 np.sum(np.isfinite(ydata.flatten())) < 100,
                                 np.sum(np.isfinite(sdata.flatten())) < 100)):
            print("Exiting: Fewer than 100 data points")
            return -1

        #stats and plot
        if mycount == 1:
            dH0 = np.copy(dH)
            dH0mean = np.nanmean(ydata)
            ydata -= dH0mean
            dH -= dH0mean
            mytitle = "DEM difference: pre-coregistration (dz0={:+.2f})".format(dH0mean)
        else:
            mytitle = "DEM difference: After Iteration {}".format(mycount - 1)
        pybob_false_hillshade(dH, mytitle, pp)
        dH_img = dH

        # calculate threshold, standard deviation of dH
        mythresh = 100 * (mystd - rmse(dH_img)) / mystd
        mystd = rmse(dH_img)

        mytitle2 = "Co-registration: Iteration {}".format(mycount)
        dx, dy, dz = pybob_nuth_fit(xdata, ydata, sdata, mytitle2, pp)
        if mycount == 1:
            dz += dH0mean
        tot_dx += dx
        tot_dy += dy
        tot_dz += dz
        magnthresh = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
        print(tot_dx, tot_dy, tot_dz)
        print(tot_dx, tot_dy, tot_dz, file=paramf)

        # shift most recent slave DEM
        #TODO: adapt this to GeoUtils
        this_slave.shift(dx, dy)  # shift in x,y
        zupdate = np.ma.array(this_slave.data + dz, mask=this_slave.mask)  # shift in z
        this_slave = this_slave.copy(new_raster=zupdate)
        this_slave = this_slave.reproject(master_dem)
        this_slave.mask(stable_mask)

        print("Percent-improvement threshold and Magnitude threshold:")
        print(mythresh, magnthresh)

    return tot_dx, tot_dy, tot_dz


def dem_coregistration(master_dem:rio.DatasetReader, slave_dem:rio.DatasetReader, inc_shp:gpd.GeoDataFrame, exc_shp=gpd.GeoDataFrame,outdir='.',method='nuth'):

    """
    DEM coregistration
    :param master_dem:
    :param slave_dem:
    :param inc_shp:
    :param exc_shp:
    :param outdir:
    :param method:
    :return:
    """

    #get start DEM statistics

    #run method: Nuth, ICP, etc...

    #get end DEM statistics

    #write results



#putting tests down here
if __name__ == '__main__':

    #TEST FOR RICHDEM SLOPE & OTHER TERRAIN ATTR WITHOUT USING GDAL
    fn_test = '/home/atom/ongoing/glaciohack_testdata/DEM_2001.64734089.tif'

    #1/ this works

    # to check it gives similar result with GDAL opening
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    ds = rio.open(fn_test)
    slp = get_terrainattr(ds,attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    #2/ this does not work (need to pass georeferencing to richDEM array, grid is not sufficient)
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))

    ds = rio.open(fn_test)
    slp = rd.TerrainAttribute(rd.rdarray(ds.read(1),no_data=ds.get_nodatavals()[0]), attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))

    #TEST FOR COREG: WAITING FOR GEOUTILS FUNCTIONS
    fn_master = ''
    fn_slave = ''
    fn_glaciermask = ''

