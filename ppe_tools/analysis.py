import os
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob

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

def calc_mean(ds,ens_name,datavar,la,cfs,units,domain='global'):

    preload = './data/'+ens_name+'_'+datavar+'_'+domain+'.nc'

    if not os.path.isdir('./data/'):
        os.system('mkdir data')
    
    #skip calculation if available on disk
    if not glob.glob(preload):  
        cf = cfs[datavar]  #conversion factor
        if cf=='intrinsic':
            if domain=='global':
                cf = 1/la.sum()/365
            else:
                cf = 1/la.groupby(ds.biome).sum()/365

        # weight by landarea
        x = la*ds[datavar]
        
        # sort out domain groupings
        x['biome']=ds.biome
        x=x.swap_dims({'gridcell':'biome'})
        if domain =='global': 
            g = 1+0*x.biome  #every gridcell is in biome 1
        else: 
            g = x.biome
        
        # calculate annual average or sum (determined by cf)
        xann = cf*(month_wts(10)*x.groupby(g).sum()).groupby('time.year').sum().compute()

        if domain =='global': 
            xann = xann.mean(dim='biome')  #get rid of gridcell dimension   

        #average/iav
        xm  = xann.mean(dim='year') 
        iav = xann.std(dim='year')

        #save the reduced data
        out = xr.Dataset()
        out[datavar+'_mean'] = xm
        out[datavar+'_mean'].attrs= {'units':units[datavar]}
        out[datavar+'_iav']  = iav
        out[datavar+'_iav'].attrs= {'units':units[datavar]}
        out['param']  = ds.param
        out['minmax'] = ds.minmax
        
        if domain=='biome':
            out['biome_name']=ds.biome_name
        
        
        out.load().to_netcdf(preload)
        
    #load from disk
    ds  = xr.open_dataset(preload)
    xm  = ds[datavar+'_mean']
    iav = ds[datavar+'_iav']
    
    return xm,iav
    

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

def get_cfs():
    '''
    loads dictionaries containing conversion factors and units
    for globally aggregating output variables  
    '''
    
    df = pd.read_csv('agg_units.csv')
    cfs   = dict()
    units = dict()
    for i,row in df.iterrows():
        f = row['field']
        u = row['unit']
        c = row['cf']

        if c != 'intrinsic':
            c = float(c)
        
        cfs[f]   = c
        units[f] = u
    return cfs,units

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
