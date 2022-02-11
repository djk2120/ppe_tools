import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob

#define the directory structure to find files
def get_files(name,htape,keys):

    topdir     = '/glade/campaign/asp/djk2120/PPEn11/hist/'
    thisdir    = topdir+name+'/'
    files      = [glob.glob(thisdir+'*'+key+'*'+htape+'*.nc')[0] for key in keys]
    return files

def ppe_init(csv='/glade/scratch/djk2120/PPEn11/surv.csv'):
    
    paramkey = pd.read_csv(csv)
    keys = paramkey.key

    #fetch the sparsegrid landarea
    la_file = '/glade/scratch/djk2120/PPEn08/sparsegrid_landarea.nc'
    la = xr.open_dataset(la_file).landarea  #km2

    #load conversion factors
    attrs = pd.read_csv('agg_units.csv',index_col=0)

    #dummy dataset
    p,m = get_params(keys,paramkey)
    ds0 = xr.Dataset()
    ds0['param']  =xr.DataArray(p,dims='ens')
    ds0['minmax'] =xr.DataArray(m,dims='ens')
    ds0['key']    =xr.DataArray(keys,dims='ens')
    whit = xr.open_dataset('./whit/whitkey.nc')
    ds0['biome']      = whit['biome']
    ds0['biome_name'] = whit['biome_name']
    
    return ds0,la,attrs,paramkey,keys

def get_ensemble(files,data_vars,keys,paramkey,p=True,extras=[]):

    def preprocess(ds):
        return ds[data_vars]

    #read in the dataset
    ds = xr.open_mfdataset(files,combine='nested',concat_dim='ens',
                           parallel=p,preprocess=preprocess)

    #diagnose htape
    htape = files[0].split('clm2.')[1].split('.')[0]
    
    #make time more sensible
    if htape=='h0' or htape=='h1':
        ds['time'] = xr.cftime_range(str(2005),periods=len(ds.time),freq='MS')
    elif htape=='h5':
        nt = len(ds.time)
        t  = ds.time.isel(time=np.arange(nt)<nt-1)
        ds = ds.isel(time=np.arange(nt)>0)
        ds['time']=t

    #specify extra variables
    if not extras:
        if htape=='h1':
            extras     = ['pfts1d_lat','pfts1d_lon','pfts1d_itype_veg']
        else:
            extras     = ['grid1d_lat','grid1d_lon']

    
    #add in some extra variables
    ds0 = xr.open_dataset(files[0])
    for extra in extras:
        ds[extra]=ds0[extra]
        
    #append some info about key/param/minmax/biome
    params,minmaxs = get_params(keys,paramkey) 
    ds['key']    = xr.DataArray(keys,dims='ens')
    ds['param']  = xr.DataArray(params,dims='ens')
    ds['minmax'] = xr.DataArray(minmaxs,dims='ens')
    whit         = xr.open_dataset('./whit/whitkey.nc')
    ds['biome']      = whit['biome']
    ds['biome_name'] = whit['biome_name']

    return ds



def get_map(da):
    '''
    Regrid from sparsegrid to standard lat/lon
    
    Better to do any dimension-reducing math before calling this function. 
    Could otherwise be pretty slow...
    '''
    
    #ACCESS the sparsegrid info
    thedir  = '/glade/u/home/forrest/ppe_representativeness/output_v4/'
    thefile = 'clusters.clm51_PPEn02ctsm51d021_2deg_GSWP3V1_leafbiomassesai_PPE3_hist.annual+sd.400.nc'
    sg = xr.open_dataset(thedir+thefile)
    
    #DIAGNOSE the shape of the output map
    newshape = []
    coords=[]
    #  grab any dimensions that arent "gridcell" from da
    for coord,nx in zip(da.coords,da.shape):
        if nx!=400:
            newshape.append(nx)
            coords.append((coord,da[coord].values))
    #  grab lat/lon from sg
    for coord in ['lat','lon']:
        nx = len(sg[coord])
        newshape.append(nx)
        coords.append((coord,sg[coord].values))

    #INSTANTIATE the outgoing array
    array = np.zeros(newshape)+np.nan
    nd    = len(array.shape)
    
    #FILL the array
    ds = xr.open_dataset('/glade/scratch/djk2120/PPEn11/hist/CTL2010/PPEn11_CTL2010_OAAT0399.clm2.h0.2005-02-01-00000.nc')
    for i in range(400):
        lat=ds.grid1d_lat[i]
        lon=ds.grid1d_lon[i]
        cc = sg.rcent.sel(lat=lat,lon=lon,method='nearest')
        ix = sg.cclass==cc
        
        
        if nd==2:
            array[ix]=da.isel(gridcell=i)
        else:
            nx = ix.sum().values
            array[:,ix]=np.tile(da.isel(gridcell=i).values[:,np.newaxis],[1,nx])
    
    #OUTPUT as DataArray
    da_map = xr.DataArray(array,name=da.name,coords=coords)
    da_map.attrs=da.attrs

    return da_map

def get_params(keys,paramkey):
    params=[]
    minmaxs=[]
    for key in keys:
        ix     = paramkey.key==key
        params.append(paramkey.param[ix].values[0])
        minmaxs.append(paramkey.minmax[ix].values[0])
    return params,minmaxs

def month_wts(nyears):
    '''
    returns an xr.DataArray of days per month, tiled for nyears
    '''
    days_pm  = [31,28,31,30,31,30,31,31,30,31,30,31]
    return xr.DataArray(np.tile(days_pm,nyears),dims='time')

def get_lapft(la,sample_h1):
    
    tmp = xr.open_dataset(sample_h1)

    nx = len(tmp.pfts1d_lat)
    pfts1d_area = np.zeros(nx)
    for i,lat,lon in zip(range(nx),tmp.pfts1d_lat,tmp.pfts1d_lon):
        ixlat = abs(lat-tmp.grid1d_lat)<0.1
        ixlon = abs(lon-tmp.grid1d_lon)<0.1
        ix    = (ixlat)&(ixlon)

        pfts1d_area[i] = la[ix]

    lapft = pfts1d_area*tmp.pfts1d_wtgcell
    lapft.name = 'patch area'
    lapft.attrs['long_name'] = 'pft patch area, within the sparsegrid'
    lapft.attrs['units'] = 'km2'
    return lapft

def get_cfs(attrs,datavar,ds,la):
    if datavar in attrs.index:
        cf1   = attrs.cf1[datavar]
        cf2   = attrs.cf2[datavar]
        if cf2=='1/lasum':
            cf2 = 1/la.sum()
        else:
            cf2 = float(cf2)
        units = attrs.units[datavar]
    else:
        cf1   = 1/365
        cf2   = 1/la.sum()
        if datavar in ds:
            units = ds[datavar].attrs['units']
        else:
            units = 'tbd'
    return cf1,cf2,units

def calc_mean(ens_name,datavar,domain='global',overwrite=False):
    '''
    Calculate the annual mean for given datavar across the ensemble.
        ens_name, one of CTL2010,CTL2010SP,AF1855,AF2095,C285,C867,NDEP
        datavar, e.g. GPP
        domain, one of global,biome,pft
        overwrite, option to rewrite existing saved data
    returns xmean,xiav
    '''
    
    ds0,la,attrs,paramkey,keys = ppe_init()
    
    preload = ('/glade/u/home/djk2120/clm5ppe/pyth/data/'+
               ens_name+'_'+datavar+'_'+domain+'.nc')
    if not glob.glob(preload):
        preload = './data/'+ens_name+'_'+datavar+'_'+domain+'.nc'
        if not os.path.isdir('./data/'):
            os.system('mkdir data')
    if overwrite:
        os.system('rm '+preload)
    
    #only calculate if not available on disk
    if not glob.glob(preload):
        longname=''
        specials = ['ALTMAX','SCA_JJA','SCA_DJF']
        
        if datavar not in specials:
            
            if domain=='pft':
                xmean,xiav,longname,units=pft_mean(ens_name,datavar,la,attrs,keys,paramkey)    
            
            if domain=='biome':
                xmean,xiav,longname,units=biome_mean(ens_name,datavar,la,attrs,keys,paramkey) 
            
            if domain=='global':
                xmean,xiav,longname,units=gcell_mean(ens_name,datavar,la,attrs,keys,paramkey) 
 
        else:
            xmean,xiav,longname = calc_special(ens_name,datavar,la)
        
        #save the reduced data
        out = xr.Dataset()
        out[datavar+'_mean'] = xmean
        out[datavar+'_mean'].attrs= {'units':units,'long_name':longname}
        out[datavar+'_iav']  = xiav
        out[datavar+'_iav'].attrs= {'units':units,'long_name':longname}
        out['param']  = ds0.param
        out['minmax'] = ds0.minmax
        if domain=='biome':
            out['biome_name']=ds0.biome_name
        if domain=='pft':
            pftkeys=['NV','NEMT','NEBT','NDBT','BETT','BEMT','BDTT','BDMT','BDBT',
                     'BES','BDMS','BDBS','C3AG','C3NG','C4G','C3C','C3I']
            out['pftkey']=xr.DataArray(pftkeys,dims='pft')
        out.load().to_netcdf(preload)

    #load from disk
    ds  = xr.open_dataset(preload)
    v   = datavar+'_iav'
    xmean   = ds[datavar+'_mean']
    if v in ds.data_vars:
        xiav = ds[v]
    else:
        xiav = []
    
    return xmean,xiav

def gcell_mean(ens,datavar,la,attrs,keys,paramkey):

    files = get_files(ens,'h0',keys)
    dvs   = datavar.split('-')
    ds    = get_ensemble(files,dvs,keys,paramkey)
    
    cf1,cf2,units = get_cfs(attrs,datavar,ds,la)
          
    x = ds[dvs[0]]
    if len(dvs)==2:
        ix = ds[dvs[1]]>0
        x  = (x/ds[dvs[1]]).where(ix).fillna(0)
        
    xann      = cf1*(month_wts(10)*x).groupby('time.year').sum()
    xann_glob = cf2*(la*xann).sum(dim='gridcell').compute()
    xmean     = xann_glob.mean(dim='year')
    xiav      = xann_glob.std(dim='year')
    longname  = ds[dvs[0]].attrs['long_name']
    return xmean,xiav,longname,units

def biome_mean(ens,datavar,la,attrs,keys,paramkey):
    files = get_files(ens,'h0',keys)
    dvs   = datavar.split('-')
    ds    = get_ensemble(files,dvs,keys,paramkey)
    
    cf1,cf2,units = get_cfs(attrs,datavar,ds,la)
    
    x = ds[dvs[0]]
    if len(dvs)==2:
        ix = ds[dvs[1]]>0
        x  = (x/ds[dvs[1]]).where(ix).fillna(0)

    xann       = cf1*(month_wts(10)*x).groupby('time.year').sum()
    xann_biome = cf2*(la*xann).groupby(ds.biome).sum().compute()
    xmean      = xann_biome.mean(dim='year')
    xiav       = xann_biome.std(dim='year')
    longname   = ds[dvs[0]].attrs['long_name']

    return xmean,xiav,longname,units

def pft_mean(ens,datavar,la,attrs,keys,paramkey):
    files = get_files(ens,'h1',keys)
    dvs   = datavar.split('-')
    ds    = get_ensemble(files,dvs,keys,paramkey)
    
    cf1,cf2,units = get_cfs(attrs,datavar,ds,la)
    lapft = get_lapft(la,files[0])
    
    x = ds[dvs[0]]
    if len(dvs)==2:
        ix = ds[dvs[1]]>0
        x  = (x/ds[dvs[1]]).where(ix).fillna(0)

    xann           = cf1*(month_wts(10)*x).groupby('time.year').sum()
    la_xann        = (lapft*xann)
    la_xann['pft'] = ds.pfts1d_itype_veg
    xann_pft       = cf2*(la_xann).groupby('pft').sum().compute()
    xmean          = xann_pft.mean(dim='year')
    xiav           = xann_pft.std(dim='year')
    longname       = ds[dvs[0]].attrs['long_name']

    return xmean,xiav,longname,units

def find_pair(da,params,minmax,p):
    '''
    returns a subset of da, corresponding to parameter-p
        the returned pair corresponds to [p_min,p_max]
    '''
    ixmin = np.logical_and(params==p,minmax=='min')
    ixmax = np.logical_and(params==p,minmax=='max')
    
    #sub in default if either is missing
    if ixmin.sum().values==0:
        ixmin = params=='default'
    if ixmax.sum().values==0:
        ixmax = params=='default'
        
    emin = da.ens.isel(ens=ixmin).values[0]
    emax = da.ens.isel(ens=ixmax).values[0]

    return da.sel(ens=[emin,emax])

def top_n(da,nx,params,minmax,uniques=[]):
    '''
    Sort for the largest perturbation effects
    
    returns lists of xmin, xmax, and the param_name for the top nx perturbations
    '''
    
    if not uniques:
        uniques = list(np.unique(params))
        if 'default' in uniques:
            uniques.remove('default')
    
    xmins=[];xmaxs=[];dxs=[]
    for u in uniques:
        pair  = find_pair(da,params,minmax,u)
        xmin  = pair[0].values
        xmax  = pair[1].values
        dx    = abs(xmax-xmin)

        xmins.append(xmin)
        xmaxs.append(xmax)
        dxs.append(dx)

    ranks = np.argsort(dxs)

    pvals = [uniques[ranks[i]] for i in range(-nx,0)]
    xmins = [xmins[ranks[i]]   for i in range(-nx,0)]
    xmaxs = [xmaxs[ranks[i]]   for i in range(-nx,0)]
    
    return xmins,xmaxs,pvals

def rank_plot(da,ds,nx,ll=True,title=None,xlabel=None):
    xmins,xmaxs,pvals = top_n(da,nx,ds.param,ds.minmax)
    xdef = da.isel(ens=0)
    plt.plot([xdef,xdef],[0,nx-1],'k:',label='default')
    plt.scatter(xmins,range(nx),marker='o',facecolors='none', edgecolors='r',label='low-val')
    plt.plot(xmaxs,range(nx),'ro',label='high-val')
    
    if ll:
        plt.legend(loc=3)
    i=-1
    for xmin,xmax in zip(xmins,xmaxs):
        i+=1
        plt.plot([xmin,xmax],[i,i],'r')
    plt.yticks(range(nx),pvals)
    if not xlabel:
        xlabel = da.name+' ['+da.attrs['units']+']'
    if not title:
        title = da.name
    plt.xlabel(xlabel)
    plt.title(title);

def brown_green():
    '''
    returns a colormap based on colorbrewer diverging brown->green
    '''

    # colorbrewer colormap, diverging, brown->green
    cmap = np.zeros([11,3]);
    cmap[0,:] = 84,48,5
    cmap[1,:] = 140,81,10
    cmap[2,:] = 191,129,45
    cmap[3,:] = 223,194,125
    cmap[4,:] = 246,232,195
    cmap[5,:] = 245,245,245
    cmap[6,:] = 199,234,229
    cmap[7,:] = 128,205,193
    cmap[8,:] = 53,151,143
    cmap[9,:] = 1,102,94
    cmap[10,:] = 0,60,48
    cmap = matplotlib.colors.ListedColormap(cmap/256)
    
    return cmap
