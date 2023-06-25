# myutils.py: Useful LSST Camera snippets and methods

import numpy as np
import pandas as pd
from lsst.obs.lsst import LsstCam
from matplotlib import pyplot as plt
from matplotlib import lines
from mpl_toolkits import axes_grid1
from matplotlib import collections
from astropy.time import Time
from astropy.io import fits
import time

import lsst.daf.butler as daf_butler
import lsst.afw.math as afw_math
from lsst.obs.lsst import LsstCam, LsstTS8
import lsst.eo.pipe as eo_pipe
from lsst.eo.pipe import (readNoiseTask, darkCurrentTask, defectsTask,
                          eperTask, divisaderoTearingTask, ptcPlotsTask,
                          linearityPlotsTask, bfAnalysisTask)


#
# utility routines for Rafts and CCDs
#

camera = LsstCam.getCamera()
det_names = {i: det.getName() for i, det in enumerate(camera)}
det_nums = {det.getName():i for i, det in enumerate(camera)}

segments = ['C%02d' % (i) for i in list(range(7+1)) + list(range(10,17+1))]
corner_bayslots = ['R00_SW0','R00_SW1','R00_SG0','R00_SG1']

def get_slots():
    slots = ['S00','S01','S02','S10','S11','S12','S20','S21','S22']
    return slots

def get_dmslots():
    dmslots = ['S20','S21','S22','S10','S11','S12','S00','S01','S02']
    return dmslots

def get_crtms():
    crtms = ['R00','R04','R40','R44']
    return crtms

def get_cslots():
    cslots = ['SG0','SG1','SW0','SW1']  #fixed to SW0,SW1
    return cslots

def get_cslots_raft():
    cslots = ['GREB0','GREB1','WREB0']  #names are different at the Raft level
    return cslots

def get_cslots_raft_TS8():
    cslots = ['GREB0','GREB1','WREB0','WREB1']
    return cslots

def get_rtms(rtm_count=21):
    if rtm_count == 9:
        rtms = ['R01','R02','R10','R11','R12','R20','R21','R22','R30']
    else:
        rtms = ['R01','R02','R03',
                'R10','R11','R12','R13','R14',
                'R20','R21','R22','R23','R24',
                'R30','R31','R32','R33','R34',
                'R41','R42','R43']

    return rtms

def get_rtmtype():
    rtm_type = {'R01':'itl','R02':'itl','R03':'itl',
                'R10':'itl','R11':'e2v','R12':'e2v','R13':'e2v','R14':'e2v',
                'R20':'itl','R21':'e2v','R22':'e2v','R23':'e2v','R24':'e2v',
                'R30':'e2v','R31':'e2v','R32':'e2v','R33':'e2v','R34':'e2v',
                'R41':'itl','R42':'itl','R43':'itl'}
    return rtm_type

#def get_allrtmtype():
#    rtm_type = {'R00':'itl','R01':'itl','R02':'itl','R03':'itl','R04':'itl',
#                'R10':'itl','R11':'e2v','R12':'e2v','R13':'e2v','R14':'e2v',
#                'R20':'itl','R21':'e2v','R22':'e2v','R23':'e2v','R24':'e2v',
#                'R30':'e2v','R31':'e2v','R32':'e2v','R33':'e2v','R34':'e2v',
#                'R40':'itl','R41':'itl','R42':'itl','R43':'itl','R44':'itl'}
#    return rtm_type

def get_allrtmtype():
    rtm_type = {'R00':'corner','R01':'itl','R02':'itl','R03':'itl','R04':'corner',
                'R10':'itl','R11':'e2v','R12':'e2v','R13':'e2v','R14':'e2v',
                'R20':'itl','R21':'e2v','R22':'e2v','R23':'e2v','R24':'e2v',
                'R30':'e2v','R31':'e2v','R32':'e2v','R33':'e2v','R34':'e2v',
                'R40':'corner','R41':'itl','R42':'itl','R43':'itl','R44':'corner'}
    return rtm_type

def get_slots_per_bay(abay,BOTnames=True):
    if abay=='R00' or abay=='R04' or abay=='R40' or abay=='R44':
        if BOTnames:
            slots = get_cslots()
        else:
            slots = get_cslots_raft_TS8()
    else:
        slots = get_slots()
    return slots

def get_raft_slot_list():
    raft_slot_list = []
    for raft in get_rtms():
        for slot in get_slots():
            raft_slot_list.append("%s_%s" % (raft,slot))

    for craft in get_crtms():
        for cslot in get_cslots():
            raft_slot_list.append("%s_%s" % (craft,cslot))

    return raft_slot_list

def get_segments(idet):
    seglist = []
    accd = camera[idet]
    for anamp in accd:
        seglist.append(anamp.getName())
    return seglist

# good runs for Science Rafts
def get_goodruns(loc='slac',useold=False):

    # could add BNL good run list, or original SLAC runs

    # these are SLAC good runs, but were not updated with the complete list of good runs post Raft rebuilding
    old_goodruns_slac = {'RTM-004':7984,'RTM-005':11852,'RTM-006':11746,'RTM-007':4576,'RTM-008':5761,'RTM-009':11415,'RTM-010':6350,\
                     'RTM-011':10861,'RTM-012':11063,'RTM-013':10982,'RTM-014':10928,'RTM-015':7653,'RTM-016':8553,'RTM-017':11166,'RTM-018':9056,'RTM-019':11808,\
                     'RTM-020':10669,'RTM-021':8988,'RTM-022':11671,'RTM-023':10517,'RTM-024':11351,'RTM-025':10722}

    # see https://confluence.slac.stanford.edu/display/LSSTCAM/List+of+Good+Runs  from 8/13/2020
    goodruns_slac = {'RTM-004':'11977','RTM-005':'11852','RTM-006':'11746','RTM-007':'11903','RTM-008':'11952','RTM-009':'11415','RTM-010':'12139',\
                     'RTM-011':'10861','RTM-012':'11063','RTM-013':'10982','RTM-014':'10928','RTM-015':'12002','RTM-016':'12027','RTM-017':'11166','RTM-018':'12120','RTM-019':'11808',\
                     'RTM-020':'10669','RTM-021':'12086','RTM-022':'11671','RTM-023':'10517','RTM-024':'11351','RTM-025':'10722',\
                     'CRTM-0002':'6611D','CRTM-0003':'10909','CRTM-0004':'11128','CRTM-0005':'11260'}

    if useold:
        goodruns = old_goodruns_slac
    else:
        goodruns = goodruns_slac

    return goodruns

def get_rtmids():
    rtmids = {'R00':'CRTM-0002','R40':'CRTM-0003','R04':'CRTM-0004','R44':'CRTM-0005',
                'R10':'RTM-023','R20':'RTM-014','R30':'RTM-012',
                'R01':'RTM-011','R11':'RTM-020','R21':'RTM-025','R31':'RTM-007','R41':'RTM-021',
                'R02':'RTM-013','R12':'RTM-009','R22':'RTM-024','R32':'RTM-015','R42':'RTM-018',
                'R03':'RTM-017','R13':'RTM-019','R23':'RTM-005','R33':'RTM-010','R43':'RTM-022',
                'R14':'RTM-006','R24':'RTM-016','R34':'RTM-008'}

    return rtmids

def get_ampseg():
    # Get segment names and index by Amp number. ordering corresponds to counting from Amp# from 1 to 16
    segmentName = {}
    ampNumber = {}
    #top
    for iAmp in range(1,8+1):
        segmentName[iAmp] = "1%d" % (iAmp - 1)
        ampNumber[segmentName[iAmp]] = iAmp
    #bottom
    for iAmp in range(9,16+1):
        segmentName[iAmp] = "0%d" % (16 - iAmp)
        ampNumber[segmentName[iAmp]] = iAmp

    return segmentName,ampNumber

def get_segment(amp):
    # Science Rafts and Guiders have amp 1-16
    # but W sensors (Slot='SW0' or 'SW1') just have amp 1-8
    # but correspondence to segment works out the same...
    # add letter C
    if amp<=8:
        segment = 'C%02d' % (amp+9) 
    else:
        segment = 'C%02d' % (16-amp)
    return segment

def get_segments(amps):
    segments = [get_segment(amp) for amp in amps]
    return segments

def get_segments_bydet(idet):
    seglist = []
    accd = camera[idet]
    for anamp in accd:
        seglist.append(anamp.getName())
    return seglist

def bayslot_segments(bayslot):
    """ return list of segments, in order, given the bayslot
    """
    segments = ['C%02d' % (i) for i in list(range(10,17+1)) + list(range(7,0-1,-1))]

    corner_bays = ['R00','R04','R40','R44']
    guiderslots = ['SG0','SG1']
    waveslots = ['SW0','SW1']
        
    bay = bayslot[0:3]
    slot = bayslot[4:]
    if bay in corner_bays:
        if slot in waveslots:
            segs = ['C%02d' % (i) for i in list(range(10,17+1))]
        elif slot in guiderslots:
            segs = segments
    else:
        segs = segments
        
    return segs

#
# Code for Image information and management
#

# get DSREFs for all images in a Run
def get_dsrefs(run,butler,detector=1):
    
    # get references to data
    #where = "exposure.science_program='%s'" % (run)
    
    where = "exposure.science_program=myrun"    
    dsrefs = list(set(butler.registry.queryDatasets('raw', where=where,  bind={"myrun": run}, detector=detector).expanded()))
    return dsrefs

# get all run info for a set of DSRefs
def get_refs_info(dsrefs):
        
    varsa = ['physical_filter','id','obs_id','exposure_time','dark_time','observation_type','observation_reason','day_obs','seq_num']
    varsb = ['mjd_begin','mjd_end']
    dfdict = {}
    
    for var in varsa:
        dfdict[var] = []
        
    for var in varsb:
        dfdict[var] = []
    
    # loop over references, get info
    for aref in dsrefs:
        
        ddict = aref.dataId.records['detector']
        dfdict['detector'] = ddict.id
        
        edict = aref.dataId.records['exposure'].toDict()
        for var in varsa:
            dfdict[var].append(edict[var])

        tbegin = edict['timespan'].begin
        tend = edict['timespan'].end
        dfdict['mjd_begin'].append(tbegin.mjd)
        dfdict['mjd_end'].append(tend.mjd)
        
    # make a Data Frame
    df = pd.DataFrame(dfdict)
    
    # sort by detector and then by time
    dfsort = df.sort_values(by=['detector','mjd_begin'])
    return dfsort

# get all run info...
def get_run_info(run,butler,detector=1):
    
    # get references to data
    #where = "exposure.science_program='%s'" % (run)
    
    where = "exposure.science_program=myrun"    
    dsrefs = list(set(butler.registry.queryDatasets('raw', where=where,  bind={"myrun": run}, detector=detector).expanded()))
        
    df = get_refs_info(dsrefs)
    
    return df


# get metadata from a set of images
def get_metadata(butler,dsrefs,keys):
    sttime = time.time()

    dfdictm = {}
    for akey in keys:
        dfdictm[akey] = []

    print("Number of images: ",len(dsrefs))
    for i,aref in enumerate(dsrefs):
        rawmeta = butler.get(aref.makeComponentRef("metadata"))
        for akey in keys:
            dfdictm[akey].append(rawmeta[akey])
        
    edtime = time.time()

    print('Total time: ',edtime-sttime)

    dfmeta = pd.DataFrame(dfdictm)
    dfmetas = dfmeta.sort_values(by=['MJD-BEG'],ignore_index=True)  #better have MJD-BEG in keys
    return dfmetas

# get metadata right from the Headers using fits IO
def get_headerdata(butler,dsrefs,keys):
    sttime = time.time()

    dfdictm = {}
    for akey in keys:
        dfdictm[akey] = []

    print("Number of images: ",len(dsrefs))
    for i,aref in enumerate(dsrefs):
        apath = butler.datastore.getURI(aref)
        hdu = fits.open(apath.path)
        header = hdu[0].header

        for akey in keys:
            dfdictm[akey].append(header[akey])
        
    edtime = time.time()

    print('Total time: ',edtime-sttime)

    dfmeta = pd.DataFrame(dfdictm)
    return dfmeta

# get eo_pipe output
def get_run_data(acq_run,weekly='w_2023_24',operator='lsstccs',repo='/repo/ir2',verbose=0):
    
    # We use the collection naming scheme to find all of the chained collections for an eo_pipe analysis of 
    # the specified run.  For our dataset queries, we'll use these collections.
    butler = daf_butler.Butler(repo)
    collections = butler.registry.queryCollections(f"u/{operator}/*_{acq_run}_{weekly}",
                                               collectionTypes=daf_butler.CollectionType.CHAINED)
    if verbose>0:
        for item in collections:
            print(item)
            
    # Several eo_pipe pipelines produce per-amp measurements for all of the CCDs in the current Camera object.
    # For LSSTCam, this would include all 205 CCDs in the focal plane; for TS8, this would be the 9 CCDs in 
    # the raft installed at that test stand (currently RTM-004).
    #
    # The `eo_pipe.get_amp_data` function will retrieve these per-amp measurements and return them as a 
    # three-level dictionary, keyed by measurement quantity, CCD (labeled by R**_S for LSSTCam), and 
    # amplifier data (labeled by channel name, e.g., 'C01').
    #
    # The '[ps]cti' quantities are measured from combined flats, so will have different values depending on the
    # filter used, since combined flats a created separately for each filter combination.
    amp_data = eo_pipe.get_amp_data(repo, collections)
    if verbose>0:
        for quantity in amp_data.keys():
            print(quantity)
            
    df_data = eopipe_DictToDfz(amp_data)
    if verbose>0:
        print(df_data.columns)
        
    return amp_data,df_data

# convert to amp_data format
def to_amp_data(df,name):
    
    outdict = {}
    for index, row in df.iterrows():
        if row.BAY_SLOT not in outdict:
            outdict[row.BAY_SLOT] = {}
            
        outdict[row.BAY_SLOT][row.SEGMENT] = row[name]
    return outdict

# get bias stability data
def get_run_biasstab(acq_run,weekly='w_2023_24',operator='lsstccs',repo='/repo/ir2',verbose=0):
    
    # We use the collection naming scheme to find all of the chained collections for an eo_pipe analysis of 
    # the specified run.  For our dataset queries, we'll use these collections.
    butler = daf_butler.Butler(repo)
    collections = butler.registry.queryCollections(f"u/{operator}/eo_*_{acq_run}_{weekly}",
                                               collectionTypes=daf_butler.CollectionType.CHAINED)    
    butler_collections = daf_butler.Butler(repo, collections=collections)
    
    if verbose>0:
        for item in collections:
            print(item)
            
    biasstab = {}
    for i in range(205):
        biasstab[i] = butler_collections.get('bias_stability_stats',detector=i)

    if verbose>0:
        key = 98
        if key in biasstab:
            biasstab[98].columns        
            df_98 = biasstab[98]
            print(tabulate(df_98[(df_98.amp_name=='C10')],headers = 'keys', tablefmt = 'psql',floatfmt='0.3f'))
            print(len(df_98[(df_98.amp_name=='C10')]))
            
    df_bias = calc_biasstability40_rms(biasstab)
    
    amp_biasdata = {}
    amp_biasdata['RMS_S'] = to_amp_data(df_bias,'RMS_S')
    amp_biasdata['RC_RMS_S'] = to_amp_data(df_bias,'RC_RMS_S')
        
    return biasstab,df_bias,amp_biasdata

#
# Display methods
#

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def drawamps(ccdtype='itl'):
    if type=='itl':
        dy = 2000
        dx = 509
    else:
        dy = 2002
        dx = 512
    
    lines = []
    lines.append([(0. - 0.5,dy - 0.5),(dx*8 -0.5,dy -0.5)])  #mid-line
    
    for iamp in range(1,8):
        lines.append([(dx*iamp - 0.5,0 - 0.5),(dx*iamp - 0.5,dy*2 - 0.5)])

    amplines=collections.LineCollection(lines,linewidth=0.5)

    return amplines


#
# Code with Raw Butlerized Images
#

def get_regions(raw0):
    """Get a dictionary of the imaging,serial and parallel sections in readout order, ie. with the Readout Node in the Lower Left""" 
    imaging = {}
    serial = {}
    parallel = {}
    det = raw0.getDetector()
    
    ccdid = det.getId()
    camdet = camera[ccdid]
    
    for seg in segments:
        
        # for raw images, get the imaging section and the serial/parallel overscan regions        
        amp = det[seg]
        rawamp = camdet[seg]
        
        serial_arr = raw0[amp.getRawSerialOverscanBBox()].getImage().array
        parallel_arr = raw0[amp.getRawParallelOverscanBBox()].getImage().array
        imaging_arr = raw0[amp.getRawDataBBox()].getImage().array

        # if need to flip X or Y, do it
        if rawamp.getRawFlipX():
            serial_arr = np.fliplr(serial_arr)
            parallel_arr = np.fliplr(parallel_arr)
            imaging_arr = np.fliplr(imaging_arr)
        if rawamp.getRawFlipY():
            serial_arr = np.flipud(serial_arr)
            parallel_arr = np.flipud(parallel_arr)
            imaging_arr = np.flipud(imaging_arr)

        imaging[seg] = imaging_arr
        serial[seg] = serial_arr
        parallel[seg] = parallel_arr
        
    return imaging,serial,parallel

def get_raw(raw0,skippre=True):
    """Get a dictionary of the raw amps in readout order, ie. with the Readout Node in the Lower Left""" 
    raw = {}
    data = {}
    serial_overscan = {}   #includes the parallel overscan region!
    parallel_overscan = {}  #only includes the data region, no pre or overscan
    
    det = raw0.getDetector()
    
    ccdid = det.getId()
    camdet = camera[ccdid]
    
    for seg in segments:
        
        # for raw images, get the imaging section and the serial/parallel overscan regions        
        amp = det[seg]
        rawamp = camdet[seg]
        
        raw_arr = raw0[amp.getRawBBox()].getImage().array

        # if need to flip X or Y, do it
        if rawamp.getRawFlipX():
            raw_arr = np.fliplr(raw_arr)
        if rawamp.getRawFlipY():
            raw_arr = np.flipud(raw_arr)

        # if desired, clip out Serial prescan
        if skippre:
            sprebb = rawamp.getRawSerialPrescanBBox()
            raw[seg] = raw_arr[:,sprebb.getMaxX()+1:]
        else:
            raw[seg] = raw_arr
                
        rawbb = rawamp.getRawBBox()
        databb = rawamp.getRawDataBBox()
        data[seg] = raw_arr[databb.getMinY():databb.getMaxY()+1,databb.getMinX():databb.getMaxX()+1]
        
        soverbb = rawamp.getRawSerialOverscanBBox()
        poverbb = rawamp.getRawParallelOverscanBBox()
        
        serial_overscan[seg] = raw_arr[rawbb.getMinY():rawbb.getMaxY()+1,soverbb.getMinX():soverbb.getMaxX()+1]
        parallel_overscan[seg] = raw_arr[poverbb.getMinY():poverbb.getMaxY()+1,poverbb.getMinX():poverbb.getMaxX()+1]
        
    return raw,data,serial_overscan,parallel_overscan

#
# Bias Stability information from the Butler
#
def calc_biasstability_rms(biasstab):

    rtmtype = get_allrtmtype()
    
    bay_slot = []
    bay_type = []
    segment = []
    rms = []
    rc_rms = []
    for idet in det_names.keys():
        bayslot_name = det_names[idet]
        bay = bayslot_name[0:3]
            
        df_bayslot = biasstab[idet]
        
        for seg in get_segments_bydet(idet):            
            df_C = df_bayslot[(df_bayslot.amp_name==seg)]
            
            bay_slot.append(bayslot_name)
            bay_type.append(rtmtype[bay])
            segment.append(seg)
            rms.append(np.std(df_C['mean']))
            rc_rms.append(np.std(df_C['rc_mean']))           

    dbias = {}
    dbias['bay_slot'] = bay_slot
    dbias['type'] = bay_type
    dbias['segment'] = segment
    dbias['rms'] = rms
    dbias['rc_rms'] = rc_rms

    # fill 
    df = pd.DataFrame(dbias)
    df.columns = df.columns.str.upper()
    return df        

def calc_biasstability40_rms(biasstab_s):

    rtmtype = get_allrtmtype()
    
    bay_slot = []
    bay_type = []
    segment = []
    rms_s = []
    rc_rms_s = []

    for idet in det_names.keys():
        bayslot_name = det_names[idet]
        bay = bayslot_name[0:3]
            
        dfs_bayslot = biasstab_s[idet]
        
        for seg in get_segments_bydet(idet):            
            dfs_C = dfs_bayslot[(dfs_bayslot.amp_name==seg)]
            
            # select 20 first and 20 last BIASes
            dfs_Csort = dfs_C.sort_values(by=['mjd'],ignore_index=True)
            maxindex = len(dfs_Csort)
            select = (dfs_Csort.index<20) | (dfs_Csort.index>maxindex-20)
            df_use = dfs_Csort[select]
            
            bay_slot.append(bayslot_name)
            bay_type.append(rtmtype[bay])
            segment.append(seg)
            
            rms_s.append(np.std(df_use['mean']))
            rc_rms_s.append(np.std(df_use['rc_mean']))           
 

    dbias = {}
    dbias['bay_slot'] = bay_slot
    dbias['type'] = bay_type
    dbias['segment'] = segment
    dbias['rms_s'] = rms_s
    dbias['rc_rms_s'] = rc_rms_s

    # fill 
    df = pd.DataFrame(dbias)
    df.columns = df.columns.str.upper()
    return df        


# 
# EoPipe Summary Information from the Butler
#
def fix_keys(amp_data):
    
    fixed_data = amp_data.copy()
    datakeys = list(amp_data.keys())
    keylut = {'SDSSi':'HIGH','SDSSi~ND_OD1.0':'LOW'}  # set these by hand
    for akey in datakeys:
        if type(akey) is tuple:
            if akey[1] in keylut:
                suffix = keylut[akey[1]]
            else:
                suffix = akey[1]
            newkey = akey[0]+"_"+suffix
            fixed_data[newkey] = fixed_data.pop(akey)

    return fixed_data

def eopipe_DictToDfz(amp_data):
    """ convert summary data from amp_data to a dataframe    
    """
    
    # some prep
    rtmtypes = get_allrtmtype()
    cornerbays = get_crtms()
    
    # here are all the data keys, need to fix the ones that are tuples
    amp_data = fix_keys(amp_data)
    datakeys = amp_data.keys()
        
    # get set of bayslot's present in any of the keys
    bayslot_set = {}
    for akey in datakeys:
        keyset = amp_data[akey]
        for bayslot in keyset:
            if type(bayslot)==int:
                print(akey,bayslot)
            #if len(bayslot)!=7:
            #    print(akey,bayslot)
        bayslot_set =  bayslot_set | amp_data[akey].keys()
    #print(bayslot_set)
    bayslot_list = sorted(list(bayslot_set))
        
    # output dictionary
    cdf = {}

    # fill lists of the data
    for akey in datakeys:
        cdf[akey] = []        
        for bayslot in bayslot_list:            
            seglist = bayslot_segments(bayslot)
            for seg in seglist:
                if bayslot in amp_data[akey]:
                    try:
                        cdf[akey].append(amp_data[akey][bayslot][seg])
                    except:
                        cdf[akey].append(np.nan)
                        #print(akey,bayslot,seg)
                else:
                    cdf[akey].append(np.nan)

    # fill lists with bay,slot,segment for the data
    allbay = []
    allslot = []
    allbayslot = []
    allamp = []      # 1 to 16 amp numbering
    allbaytype = []  # S or C
    allrtmtype = []  # itl or e2v
    allsegment = []  # Cxx 
    for bayslot in bayslot_list:
        seglist = bayslot_segments(bayslot)
        for i,seg in enumerate(seglist):
            allbayslot.append(bayslot)
            allbay.append(bayslot[0:3])
            allslot.append(bayslot[4:])
            allamp.append(i)
            allsegment.append(seg)
            allrtmtype.append(rtmtypes[bayslot[0:3]])
            if bayslot[0:4] in cornerbays:
                allbaytype.append('C')
            else:
                allbaytype.append('S')
            
            
    # add to DF with ccd info
    cdf['BAY'] = allbay
    cdf['SLOT'] = allslot
    cdf['AMP'] = allamp
    cdf['BAYTYPE'] = allbaytype
    cdf['BAY_SLOT'] = allbayslot
    cdf['SEGMENT'] = allsegment
            
        
    # fill 
    df = pd.DataFrame(cdf)
    df.columns = df.columns.str.upper()
    return df




# Plotting methods

def compare_tworuns(df1,df2,run1,run2,minxy,maxxy,quantity='READ_NOISE',draw_line=True,legend_loc='lower right',scale='linear',save=None):

    rtms = get_rtms()
    crtms = get_crtms()
    rtmids = get_rtmids()
    allrtms = rtms + crtms

    f,ax = plt.subplots(5,5,figsize=(22,22),constrained_layout=True)
    axf = ax

    for i,abay in enumerate(allrtms):

        thertm = rtmids[abay]
        ix = 4-int(abay[1:2])
        iy = int(abay[2:3])

        # get the desired quantity, filtered by raft
        df1f = df1[df1.BAY==abay]
        df2f = df2[df2.BAY==abay]

        for aslot in get_slots_per_bay(abay):

            # filter by slot
            df1fs = df1f[df1f.SLOT==aslot]
            df2fs = df2f[df2f.SLOT==aslot]

            quant1 = df1fs[quantity]
            quant2 = df2fs[quantity]

            # make sure we have entries
            if len(quant1)>0 and len(quant1)==len(quant2):
                axf[ix,iy].scatter(quant1,quant2,label=aslot)


        axf[ix,iy].text(0.07,0.9,'%s %s' % (abay,thertm),transform=axf[ix,iy].transAxes)
        axf[ix,iy].set_xlabel('%s Run %s' % (quantity,run1))
        axf[ix,iy].set_ylabel('%s Run %s' % (quantity,run2))

        axf[ix,iy].set_xlim(minxy,maxxy)
        axf[ix,iy].set_ylim(minxy,maxxy)

        ax[ix,iy].set_xscale(scale)
        ax[ix,iy].set_yscale(scale)

        if (ix==0 and iy==0) or (ix==0 and iy==1):
            axf[ix,iy].legend(loc=legend_loc)

        if draw_line:
            line = lines.Line2D([minxy,maxxy], [minxy,maxxy], lw=2., color='r', alpha=0.4)
            axf[ix,iy].add_line(line)
            
    if save:
        f.savefig(save)
