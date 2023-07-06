import os
import glob
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import lines
from mpl_toolkits import axes_grid1
from astropy.visualization import imshow_norm, MinMaxInterval,AsinhStretch,LinearStretch,SqrtStretch,ZScaleInterval,AsymmetricPercentileInterval,ManualInterval
from astropy import stats
from astropy.io import fits
from astropy.table import Table
import lsst.eotest.image_utils as imutils
import lsst.afw.image as afwImage
from lsst.eotest.sensor.MaskedCCD import MaskedCCD
from lsst.eotest.sensor.AmplifierGeometry import parse_geom_kwd
import lsst.eotest.sensor as sensorTest

# Dubois eTraveler DB code
from exploreRaft import exploreRaft
#from get_EO_analysis_results import get_EO_analysis_results    # uncomment if using a very old release, or if Richard updates MutableMapping
#from get_EO_analysis_files import get_EO_analysis_files


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

def get_allrtmtype():
    rtm_type = {'R00':'itl','R01':'itl','R02':'itl','R03':'itl','R04':'itl',
                'R10':'itl','R11':'e2v','R12':'e2v','R13':'e2v','R14':'e2v',
                'R20':'itl','R21':'e2v','R22':'e2v','R23':'e2v','R24':'e2v',
                'R30':'e2v','R31':'e2v','R32':'e2v','R33':'e2v','R34':'e2v',
                'R40':'itl','R41':'itl','R42':'itl','R43':'itl','R44':'itl'}
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
    if amp<=8:
        segment = '%02d' % (amp+9) 
    else:
        segment = '%02d' % (16-amp)
    return segment

def get_segments(amps):
    segments = [get_segment(amp) for amp in amps]
    return segments


def getccdid(raft,run=None,db='Prod'):
    ccdid = {}
    eR = exploreRaft(db=db)
    ccd_list = eR.raftContents(raftName=raft,run=run)
    for araft in ccd_list:
        ccdid[araft[1]] = araft[0]
    return ccdid

def getrafttype(raft,db='Prod'):
    ccdid = {}
    eR = exploreRaft(db=db)
    raft_type = eR.raft_type(raft=raft)
    return raft_type

def getbaytype(bay):
    if bay=='R00' or bay=='R40' or bay=='R04' or bay=='R44':
        baytype = 'C'
    else:
        baytype = 'S'
    return baytype

def get_amps_perbayslot(bayslot):
    bay = bayslot[0:3]
    slot = bayslot[4:7]
    if slot == 'SW0' or slot=='SW1':
        namps = 8
    else:
        namps = 16
    return namps

# A function to help gather the data.
def glob_files(root_dir,raft,run,*args):
    apath = os.path.join(root_dir, raft, run, *args)
    #print(apath)
    return sorted(glob.glob(apath))

# A function to help gather the data.
def glob_files_run(root_dir,run,*args):
    apath = os.path.join(root_dir, run, *args)
    #print(apath)
    return sorted(glob.glob(apath))


# RTM information

def getEOFiles(filesuffix='median_sflat.fits',acqtype='dark_defects_raft',rtmid='RTM-009',run=11415,disk='fs3'):

    # Specify a raft and a run to analyze.
    root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-11021_RTM' % (disk)
    raft = 'LCA-11021_%s' % (rtmid)
    run = str(run)

    # get CCDs by slot.
    ccdid = getccdid(raft,run=run)

    # loop over slots
    eofiles = {}
    for aslot in ccdid:
        eofiles[aslot] = glob_files(root_dir,raft,run,acqtype, 'v0', '*','%s_%s' % (ccdid[aslot],filesuffix) )

    return eofiles

def getEOacqFiles(filetype='dark_bias',acqtype='dark_raft_acq',rtmid='RTM-009',run=11415):

    disks = ['fs1','fs3']
    eofiles = {}

    for adisk in disks:
        # Specify a raft and a run to analyze.
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-11021_RTM' % (adisk)
        raft = 'LCA-11021_%s' % (rtmid)
        run = str(run)

        # get CCDs by slot.
        ccdid = getccdid(raft,run=run)

        # loop over slots
        for aslot in ccdid:
            filelist = sorted(glob_files(root_dir,raft,run,acqtype, 'v0', '*',aslot,'*%s*' % (filetype) ))
            if aslot in eofiles.keys():
                eofiles[aslot].extend(filelist)
            else:
                eofiles[aslot] = filelist

    return eofiles

# get summary data from one raft,run from TS8
def getEOInfo(rtmid='RTM-009',run=11415):

    disks = ['fs1','fs3']
    raft_results = {}

    for adisk in disks:

        # Specify a raft and a run to analyze.
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-11021_RTM' % (adisk)
        raft = 'LCA-11021_%s' % (rtmid)
        run = str(run)

        # get CCDs by slot.
        ccdid = getccdid(raft)

        # For linearity and ptc tasks.
        for aslot in ccdid:
            filelist = glob_files(root_dir,raft,run,'collect_raft_results', 'v0', '*','%s_eotest_results.fits' % (ccdid[aslot]))
            if aslot in raft_results.keys():
                raft_results[aslot].extend(filelist)
            else:
                raft_results[aslot] = filelist

    # put results into one big DataFrame
    dfList = []
    for aslot in ccdid:
        hdu = fits.open(raft_results[aslot][0])
        table = Table(hdu[1].data)
        df = table.to_pandas()

        # add to DF with more info
        df['SLOT'] = aslot
        df['CCDID'] = ccdid[aslot]

        dfList.append(df)

    # combine DFs
    dfAll = pd.concat(dfList, ignore_index=True)

    return dfAll,ccdid


# get summary data from one Run & Raft, package in a DataFrame
def getDBinfo(run='11415',site='I&T-Raft',type='S'):

    if run[-1:]=='D':
        runnum = int(run[0:-1])
        db= 'Dev'
        server = 'Dev'
    else:
        runnum = int(run)
        db = 'Prod'
        server = 'Prod'

    # get eTraveler DB access
    g = get_EO_analysis_results(db=db, server=server)

    # only need to specify the Run number
    raft_list_all, data_all = g.get_tests(site_type=site, run=runnum)

    res_all = g.get_all_results(data=data_all, device=raft_list_all)

    # remove some keys that aren't by Amplifier, or that aren't actually filled
    res_all.pop('QE',None)
    res_all.pop('detections',None)
    res_all.pop('band',None)
    keylist=[]
    for key in res_all.keys():
        keylist.append(key)
    for key in keylist:
        if 'max_deviation' in key:
            res_all.pop(key)

    # convert to a DataFrame
    # all but QE are per amp, QE is per wavelength so skip it
    cdf = pd.DataFrame()

    if type=='S':
        slots = get_slots()
    else:
        slots = get_cslots_raft_TS8()

    # loop over all DB data
    for akey in res_all:
        allval = []
        for islot,ccdserial in enumerate(res_all[akey]):
            vals = res_all[akey][ccdserial]

            # there is a bug in WREB0 values - some are doubled to 32 entries, instead of just 16
            if ccdserial=='WREB0':
                vals = vals[0:16]
            elif ccdserial=='WREB1':
                if len(vals)!=8:
                    print("WREB1 has only ",len(vals)," values for ",akey)
                    nmore = 8 - len(vals)
                    vals.extend(nmore*[0.0])
            allval.extend(vals)

        if len(allval)>0 :
            cdf[akey.upper()] = allval

    # loop over data once more, this time to fill slot and ccdserial numbers (use lost akey value)
    allccd = []
    allslot = []
    allamp = []
    for islot,ccdserial in enumerate(res_all[akey]):
        namps = 16
        if slots[islot]=='WREB0' or slots[islot]=='WREB1' :
            namps = 8

        ccdl =  [ccdserial] * namps
        slotl = [slots[islot]] * namps
        ampl = range(1,namps+1)
        allccd.extend(ccdl)
        allslot.extend(slotl)
        allamp.extend(ampl)

    # add to DF with ccd info
    cdf['CCDID'] = allccd
    cdf['SLOT'] = allslot
    cdf['AMP'] = allamp

    return cdf


# BOT information



def getDBinfoBOT(run='6935D',site='I&T-BOT'):
    #get summary data from one Run for the BOT, package in a DataFrame

    if run[-1:]=='D':
        runnum = int(run[0:-1])
        db= 'Dev'
    else:
        runnum = int(run)
        db = 'Prod'

    #print(runnum,db)

    # get eTraveler DB access
    g = get_EO_analysis_results(db=db, server=db)

    # only need to specify the Run number
    raft_list_all, data_all = g.get_tests(site_type=site, run=runnum)

    res_all = g.get_all_results(data=data_all, device=raft_list_all)

    # convert to a DataFrame
    # all but QE are per amp, QE is per wavelength so skip it
    cdf = pd.DataFrame()

    # loop over all DB data
    for akey in res_all:
        allval = []
        for ibay,bay in enumerate(res_all[akey]):
            for islot,slot in enumerate(res_all[akey][bay]):
                vals = res_all[akey][bay][slot]
                allval.extend(vals)

        # QE is only recorded per CCD, so skip that if present, 
        # also sometimes Tearing detections are also not present for all Amps, catch that 
        if (akey != 'QE'):
            if len(allval)>0 :
                try:
                    cdf[akey.upper()] = allval
                except:
                    print(akey,len(allval))       

    # loop over data once more, this time to fill bay, slot and amp numbers (use last akey value)
    allbay = []
    allslot = []
    allamp = []
    allbaytype = []
    allsegment = []
    akey = list(res_all.keys())[0]
    for bay in res_all[akey]:
        for slot in res_all[akey][bay]:

#            if slot[0:2] == 'SW':
#                namp = 8
#            else:
            namp = 16

            bayl =  [bay] * namp
            slotl = [slot] * namp
            baytypel = getbaytype(bay) * namp
            ampl = range(1,namp+1)
            segmentl = get_segments(ampl)
            allbay.extend(bayl)
            allslot.extend(slotl)
            allamp.extend(ampl)
            allbaytype.extend(baytypel)
            allsegment.extend(segmentl)

    # add to DF with ccd info
    cdf['BAY'] = allbay
    cdf['SLOT'] = allslot
    cdf['AMP'] = allamp
    cdf['BAYTYPE'] = allbaytype
    cdf['BAY_SLOT'] = cdf.BAY + "_" + cdf.SLOT
    cdf['SEGMENT'] = allsegment
    
    # add TYPE, 'e2v' or 'itl'
    manu = get_allrtmtype()
    manufact = [manu[bay] for bay in cdf.BAY]
    cdf["TYPE"] = manufact

    # SW slots have 8 extra entries now, need to remove...
    bad = cdf[(((cdf['SLOT']=='SW0') | (cdf['SLOT']=='SW1')) & (cdf['AMP']>=9))].index
    cdf.drop(bad,inplace=True)
    
    # also reset the index! 
    cdf = cdf.reset_index()

    return cdf


def getBOTEOInfo(run="11415",disk='fs3',rtm_count=21):

    # Specify a raft and a run to analyze.
    if run[-1:]=='D':
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
    else:
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
    run = str(run)

    # Get all result files
    all_results = glob_files_run(root_dir,run,'raft_results_summary_BOT', 'v0', '*','*_eotest_results.fits')
    #print(all_results)

    # identify which files correspond to which rafts/slots
    crtms = get_crtms()
    rtms = get_rtms(rtm_count)

    cslots = get_cslots()
    slots = get_slots()

    bayslots = []
    results = {}
    for abay in rtms:
        for aslot in slots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    for abay in crtms:
        for aslot in cslots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    #print(bayslots)

    for aresult in all_results:
        for bayslot in bayslots:
            if bayslot in aresult:
                results[bayslot] = aresult

    #print(results)
    # put results into one big DataFrame
    dfList = []
    for bayslot in bayslots:
        if bayslot in results.keys():
            hdu = fits.open(results[bayslot])
            table = Table(hdu[1].data)
            if bayslot[4:6]=='SW':   # special for SW - they only have 8 amps, but tables have 16
                table = table[0:8]
            df = table.to_pandas()

            # add to DF with more info
            df['BAY_SLOT'] = bayslot
            df['BAY'] = bayslot[0:3]
            df['SLOT'] = bayslot[4:]
            df['BAYTYPE'] = getbaytype(bayslot[0:3])
            #df['CCDID'] = ccdid[aslot]

            dfList.append(df)

    # combine DFs
    dfAll = pd.concat(dfList, ignore_index=True,sort=False)

    return dfAll

# get flat pair summary data into a DataFrame
def getBOTFlatPairs(run='6846D',analysis_dir='flat_pairs_BOT',file_suffix='det_response.fits',disk='fs3',rtm_count=9):

    # Specify a raft and a run to analyze.
    if run[-1:]=='D':
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
    else:
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)

    run = str(run)

    # Get all result files
    files = glob_files_run(root_dir,run,analysis_dir, 'v0', '*','*_%s' % (file_suffix))


    # identify which files correspond to which rafts/slots
    crtms = get_crtms()
    rtms = get_rtms(rtm_count)

    cslots = get_cslots()
    slots = get_slots()

    bayslots = []
    results = {}
    for abay in rtms:
        for aslot in slots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    for abay in crtms:
        for aslot in cslots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    # find file for each bayslot
    for afile in files:
        for bayslot in bayslots:
            if bayslot in afile:
                results[bayslot] = afile

    #print(results)
    # put results into one big DataFrame
    dfList = []
    for bayslot in bayslots:
        if bayslot in results.keys():

            try:
                hdu = fits.open(results[bayslot])
                table = Table(hdu[1].data)
                df = table.to_pandas()

                # add to DF with more info
                df['BAYSLOT'] = bayslot
                df['BAY'] = bayslot[0:3]
                df['SLOT'] = bayslot[4:]
                #df['CCDID'] = ccdid[aslot]

                dfList.append(df)
            except Exception as e:
                print(e)
                print(bayslot)
                print(results[bayslot])

    # combine DFs
    dfAll = pd.concat(dfList, ignore_index=True,sort=False)

    return dfAll

def getBOTEOFiles(run="11415",type='overscan_results',typedir='overscan',disk='fs3',location='archive',rtm_count=21):

    # Specify a raft and a run to analyze.
    if run[-1:]=='D':
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_%s-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk,location)
    else:
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_%s/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk,location)
    run = str(run)

    # Get all result files
    all_results = glob_files_run(root_dir,run,'%s_BOT' % (typedir), 'v0', '*','*_%s.fits' % (type))
    #print(all_results)

    # identify which files correspond to which rafts/slots
    crtms = get_crtms()
    rtms = get_rtms(rtm_count)

    cslots = get_cslots()
    slots = get_slots()

    bayslots = []
    results = {}
    for abay in rtms:
        for aslot in slots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    for abay in crtms:
        for aslot in cslots:
            bayslot = '%s_%s' % (abay,aslot)
            bayslots.append(bayslot)

    # now get a dictionary of files, keyed by bayslot Rxx_Sxx
    filedict = {}
    for bayslot in bayslots:
        files = glob_files_run(root_dir,run,'%s_BOT' % (typedir), 'v0', '*','%s_*_%s.fits' % (bayslot,type))
        filedict[bayslot] = files

    return all_results,filedict

def getBOTacqlinks(run='6806D',disk='fs3',archive=True):
    # get BOT data files
    file_links = {}

    # Specify a raft and a run to analyze.
    if archive:
        if run[-1:]=='D':
            root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
        else:
            root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
    else:
        if run[-1:]=='D':
            root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_stage-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
        else:
            root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_stage/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)

    # run number
    srun = str(run)

    # A function to help gather the data.
    def glob_files(*args):
        apath = os.path.join(root_dir, srun, 'BOT_acq','v0','*',*args)
        files = sorted(glob.glob(apath))
        return files

    # image types
    acqtypes = {'bias':['bias'],'dark':['bias','dark'],'flat':['bias','flat'],'lambda':['flat'],'sflat':['L','H'],'fe55':['flat']}

    for atype in acqtypes:
        imgtypes = acqtypes[atype]
        for aimg in imgtypes:
            akey = atype + '_' + aimg
            file_links[akey] = glob_files('%s*%s*' % (atype,aimg))

    return file_links

# get links to analysis fits files, keyed by type, raft and slot
def getBOTanalinks(run='6806D',disk='fs3'):
    # get BOT data files
    file_links = {}

    # Specify a raft and a run to analyze.
    if run[-1:]=='D':
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive-test/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)
    else:
        root_dir = '/gpfs/slac/lsst/%s/g/data/jobHarness/jh_archive/LCA-10134_Cryostat/LCA-10134_Cryostat-0001/' % (disk)

    # run number
    srun = str(run)

    # A function to help gather the data.
    def glob_files(anadir,*args):
        apath = os.path.join(root_dir, srun, '%s_BOT' % (anadir),'v0','*',*args)
        files = sorted(glob.glob(apath))
        return files

    # image types
    anatypes = {'bias_frame':['median_bias'],'pixel_defects':['median_dark','median_sflat'],'cti':['superflat_low','superflat_high']}

    # build dictionaries for each image type
    for atype in anatypes:
        imgtypes = anatypes[atype]
        for aimg in imgtypes:
            akey = atype + '_' + aimg
            file_links[akey] = {}

    # get list of raft_slot
    rslist = get_raft_slot_list()

    for atype in anatypes:
        imgtypes = anatypes[atype]
        for aimg in imgtypes:
            akey = atype + '_' + aimg
            for raft_slot in rslist:
                files = glob_files(atype,'*%s*%s*.fits' % (raft_slot,aimg))
                if len(files)>0:
                    file_links[akey][raft_slot] = files[0]
                else:
                    file_links[akey][raft_slot] = None

    return file_links


def getBOTimgfiles(file_links,raft,slot):

    outdict = {}

    # build file names for a given set of links and a single raft, slot
    for alink in file_links:
        all_files = []
        for link in file_links[alink]:
            apath = os.path.join(link,'*_%s_%s.fits' % (raft,slot))
            files = sorted(glob.glob(apath))
            all_files += files
        outdict[alink] = all_files

    return outdict


# curated gains
def get_curated_gains(curated_gain_file):

    gaindict = json.load(open(curated_gain_file,'r'))

    # convert json Gain file to a dataframe with BAY,SLOT,AMP,PTC_GAIN with amps in 1 to 16 order...
    bay_list = []
    slot_list = []
    amp_list = []
    gain_list = []

    for key in gaindict.keys():
        adict = gaindict[key]
        bay = key[0:3]
        slot = key[4:]
        for i,segment in enumerate(adict.keys()):
            amp = i
            gain = adict[segment]
        
            bay_list.append(bay)
            slot_list.append(slot)
            amp_list.append(amp+1)  # make sure its 1-16
            gain_list.append(gain)
        
    outdict = {'BAY':bay_list,'SLOT':slot_list,'AMP':amp_list,'PTC_GAIN':gain_list}
    df_gain = pd.DataFrame(outdict)
    return df_gain


# analysis methods

def normed_mean_response_vscol(sflat_file,sensor_type):
    # for an input .fits file, calculates the normalized sigma clipped mean flux vs. Col# for a group of Rows
    # returns two arrays for the top and bottom section of the CCD
    ncol_itl = 509
    ncol_e2v = 512
    ncol_dict = {'itl':ncol_itl,'e2v':ncol_e2v}

    amc = sensorTest.MaskedCCD(sflat_file)
    imaging = amc.amp_geom.imaging
    # use 200 rows close to the amplifier
    row_lo = 10
    row_hi = 210

    # top row
    averow_top = np.zeros((ncol_dict[sensor_type]*8))
    for iAmp in range(1,8+1):
        # Segments 10-17
        anamp = imutils.trim(amc[iAmp],imaging=imaging)
        anamp_im = anamp.getImage()
        anamp_arr = anamp_im.getArray()

        # use a robust mean
        anamp_meanbyrow,medbyrow,stdbyrow = stats.sigma_clipped_stats(anamp_arr[row_lo:row_hi,:],axis=0)

        # normalize
        nmean_byrow = anamp_meanbyrow/np.median(anamp_meanbyrow)

        lopix = 0 + (iAmp-1)*ncol_dict[sensor_type]
        hipix = ncol_dict[sensor_type] + (iAmp-1)*ncol_dict[sensor_type]
        averow_top[lopix:hipix] = np.flip(nmean_byrow)

    # bot row
    averow_bot = np.zeros((ncol_dict[sensor_type]*8))
    for jAmp in range(16,8,-1):
        # Segments 00-07
        iAmp = 17 - jAmp # iAmp goes from 1 to 8, in order of increasing Yccs
        anamp = imutils.trim(amc[jAmp],imaging=imaging)
        anamp_im = anamp.getImage()
        anamp_arr = anamp_im.getArray()

        # use a robust mean
        anamp_meanbyrow,medbyrow,stdbyrow = stats.sigma_clipped_stats(anamp_arr[row_lo:row_hi,:],axis=0)

        # normalize
        nmean_byrow = anamp_meanbyrow/np.median(anamp_meanbyrow)

        lopix = 0 + (iAmp-1)*ncol_dict[sensor_type]
        hipix = ncol_dict[sensor_type] + (iAmp-1)*ncol_dict[sensor_type]
        if sensor_type=='e2v':
            averow_bot[lopix:hipix] = nmean_byrow
        elif sensor_type=='itl':
            averow_bot[lopix:hipix] = np.flip(nmean_byrow)

    # analyze the gaps between amplifiers for Divisidero Tearing, and find the max(abs) deviation in the +-2 columns at the boundaries
    max_divisidero_tearing = []    # 14 entries per CCD
    for k in range(1,7+1):
        collo = ncol_dict[sensor_type]*k - 2  # 2nd to last column in Amplifier
        max_divisidero = np.max(np.abs(averow_top[collo:collo+4] - 1.0))    # +-2 columns
        max_divisidero_tearing.append(max_divisidero)
    for k in range(1,7+1):
        collo = ncol_dict[sensor_type]*k - 2  # 2nd to last column in Amplifier
        max_divisidero = np.max(np.abs(averow_bot[collo:collo+4] - 1.0))    # +-2 columns
        max_divisidero_tearing.append(max_divisidero)

    return averow_top,averow_bot,max_divisidero_tearing

def ana_divisidero_tearing(run,rtmid):
    # for a given run, raft combination analyzes a corrected super-flat for Divisidero Tearing

    ncol_itl = 509
    ncol_e2v = 512
    ncol_dict = {'itl':ncol_itl,'e2v':ncol_e2v}

    sflat_files = getEOFiles(rtmid=rtmid,run=run)
    sensor_type = getrafttype('LCA-11021_%s' % (rtmid))
    sensor_type = sensor_type.lower()
    ccdid = getccdid('LCA-11021_%s' % (rtmid))

    # make x pixel values
    xpixval = np.arange(ncol_dict[sensor_type]*8)

    # get row averages
    avedict = {}
    for slot in ccdid:
        avedict[slot] = normed_mean_response_vscol(sflat_files[slot][0],sensor_type)

    # make a summary plot
    f  = plt.figure(figsize=(20,20))
    outer = gridspec.GridSpec(3,3,wspace=0.3,hspace=0.3)

    # dmslotorder
    dmslots = get_dmslots()

    nskip_edge = 20

    for i,slot in enumerate(dmslots):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.0)

        for j in range(2):

            # use max of max_divisidero_tearing to set the range of plots
            max_divisidero = avedict[slot][2]
            plot_range = np.max(max_divisidero[j*7:j*7+8])

            ax = plt.Subplot(f,inner[j])
            ax.plot(xpixval[nskip_edge:ncol_dict[sensor_type]*8 - nskip_edge],avedict[slot][j][nskip_edge:ncol_dict[sensor_type]*8 - nskip_edge])
            ax.set_xlabel('Col #')
            ax.set_ylim(1.-plot_range,1.+plot_range)
            for k in range(1,8):
                ax.axvline(x=ncol_dict[sensor_type]*k,color='red',ls='--',alpha=0.2)
            if j==0:
                t = ax.text(0.025,0.9, '%s' % (slot),transform = ax.transAxes)
                t = ax.text(0.825,0.05, 'Seg 10-17',transform = ax.transAxes)
            elif j==1:
                t = ax.text(0.825,0.05, 'Seg 00-07',transform = ax.transAxes)

            f.add_subplot(ax)

    plt.suptitle('Run %s %s' % (str(run),rtmid),fontsize=36)

    return avedict

##### Image making routines

def make_ccd_mosaic(infile, bias_frame=None, gains=None, fit_order=1,dm_view=False):
    """Combine amplifier image arrays into a single mosaic CCD image array."""
    ccd = MaskedCCD(infile, bias_frame=bias_frame)
    datasec = parse_geom_kwd(ccd.amp_geom[1]['DATASEC'])
    nx_segments = 8
    ny_segments = 2
    nx = nx_segments*(datasec['xmax'] - datasec['xmin'] + 1)
    ny = ny_segments*(datasec['ymax'] - datasec['ymin'] + 1)
    mosaic = np.zeros((ny, nx), dtype=np.float32) # this array has [0,0] in the upper right corner on LCA-13381 view of CCDs and [ny,nx] in the lower right

    for ypos in range(ny_segments):
        for xpos in range(nx_segments):
            amp = ypos*nx_segments + xpos + 1

            detsec = parse_geom_kwd(ccd.amp_geom[amp]['DETSEC'])
            xmin = nx - max(detsec['xmin'], detsec['xmax'])
            xmax = nx - min(detsec['xmin'], detsec['xmax']) + 1
            ymin = ny - max(detsec['ymin'], detsec['ymax'])
            ymax = ny - min(detsec['ymin'], detsec['ymax']) + 1
            #
            # Extract bias-subtracted image for this segment - overscan not corrected, since we don't pass overscan here(?)
            #
            segment_image = ccd.unbiased_and_trimmed_image(amp, fit_order=fit_order)
            subarr = segment_image.getImage().getArray()
            #
            # Determine flips in x- and y- direction
            #
            if detsec['xmax'] > detsec['xmin']: # flip in x-direction
                subarr = subarr[:, ::-1]
            if detsec['ymax'] > detsec['ymin']: # flip in y-direction
                subarr = subarr[::-1, :]
            #
            # Convert from ADU to e-
            #
            if gains is not None:
                subarr *= gains[amp]
            #
            # Set sub-array to the mosaiced image
            #
            mosaic[ymin:ymax, xmin:xmax] = subarr

    if dm_view:
        # transpose and rotate by -90 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower') rotated CW by 90 for DM view
        mosaicprime = np.zeros((ny, nx), dtype=np.float32)
        mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=-1)

        image = afwImage.ImageF(mosaicprime)

    else:
        # transpose and rotate by 180 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower')
        mosaicprime = np.zeros((nx, ny), dtype=np.float32)
        mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=2)

        image = afwImage.ImageF(mosaicprime)

    return image


# new version with row,col overscan option
def make_ccd_mosaic_rowcol(infile, bias_frame=None, gains=None, fit_order=1,dm_view=False,bias_method='rowcol'):
    """Combine amplifier image arrays into a single mosaic CCD image array."""
    if bias_frame==None:
        ccd = MaskedCCD(infile)
    else:        
        ccd = MaskedCCD(infile, bias_frame=bias_frame)
        
    datasec = parse_geom_kwd(ccd.amp_geom[1]['DATASEC'])
    nx_segments = 8
    ny_segments = 2
    nx = nx_segments*(datasec['xmax'] - datasec['xmin'] + 1)
    ny = ny_segments*(datasec['ymax'] - datasec['ymin'] + 1)
    mosaic = np.zeros((ny, nx), dtype=np.float32) # this array has [0,0] in the upper right corner on LCA-13381 view of CCDs and [ny,nx] in the lower right

    for ypos in range(ny_segments):
        for xpos in range(nx_segments):
            amp = ypos*nx_segments + xpos + 1

            detsec = parse_geom_kwd(ccd.amp_geom[amp]['DETSEC'])
            xmin = nx - max(detsec['xmin'], detsec['xmax'])
            xmax = nx - min(detsec['xmin'], detsec['xmax']) + 1
            ymin = ny - max(detsec['ymin'], detsec['ymax'])
            ymax = ny - min(detsec['ymin'], detsec['ymax']) + 1
            #
            # Extract bias-subtracted image for this segment - overscan not corrected, since we don't pass overscan here(?)
            #
            if bias_frame==None:
                
                # Make a copy of the raw amp image
                segment = ccd[amp].Factory(ccd[amp], deep=True)
                
                # Subtract the parallel+serial overscan-based bias image
                segment -= ccd.bias_image_using_overscan(amp, bias_method=bias_method)
                
                # now get just the trimmed image
                segment_image = imutils.trim(segment, ccd.amp_geom.imaging)

            else:
                segment_image = ccd.unbiased_and_trimmed_image(amp, fit_order=fit_order)
                
            subarr = segment_image.getImage().getArray()
            #
            # Determine flips in x- and y- direction
            #
            if detsec['xmax'] > detsec['xmin']: # flip in x-direction
                subarr = subarr[:, ::-1]
            if detsec['ymax'] > detsec['ymin']: # flip in y-direction
                subarr = subarr[::-1, :]
            #
            # Convert from ADU to e-
            #
            if gains is not None:
                subarr *= gains[amp]
            #
            # Set sub-array to the mosaiced image
            #
            mosaic[ymin:ymax, xmin:xmax] = subarr

    if dm_view:
        # transpose and rotate by -90 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower') rotated CW by 90 for DM view
        mosaicprime = np.zeros((ny, nx), dtype=np.float32)
        mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=-1)

        image = afwImage.ImageF(mosaicprime)

    else:
        # transpose and rotate by 180 to get a mosaic ndarray that will look like the LCA-13381 view with matplotlib(origin='lower')
        mosaicprime = np.zeros((nx, ny), dtype=np.float32)
        mosaicprime[:,:] = np.rot90(np.transpose(mosaic),k=2)

        image = afwImage.ImageF(mosaicprime)

    return image

##### Plotting routines

def mkProfile(xarr,yarr,nx=100,xmin=0.,xmax=1.0,ymin=0.,ymax=1.0,retPlot=True):
    dx = (xmax-xmin)/nx
    bins = np.arange(xmin,xmax+dx,dx)
    nbin = len(bins)-1
    #print(dx,bins,nbin)
    inrange = (yarr>=ymin) & (yarr<ymax)
    yinrange = yarr[inrange]
    xinrange = xarr[inrange]
    ind = np.digitize(xinrange,bins) - 1.   #np.digitize starts at bin=1
    xval = np.zeros(nbin)
    xerr = np.zeros(nbin)
    yval = np.zeros(nbin)
    yerr = np.zeros(nbin)
    for i in range(nbin):
        inbin = (ind==i)
        xinbin = xinrange[inbin]
        yinbin = yinrange[inbin]
        nentries = len(yinbin)
        xval[i] = 0.5*(bins[i+1]+bins[i])
        xerr[i] = 0.5*(bins[i+1]-bins[i])
        if nentries>0:
            yval[i] = np.mean(yinbin)
            yerr[i] = np.std(yinbin)/np.sqrt(nentries)
            #print(i,xval[i],xerr[i],yval[i],yerr[i])
    if retPlot:
        profile = plt.errorbar(xval,yval,xerr=xerr,yerr=yerr)
        return profile
    else:
        return xval,yval,xerr,yerr

# loop over bays, compare READ_NOISE in two BOT runs
def compare_tworuns(df1,df2,run1,run2,minxy,maxxy,quantity='READ_NOISE',draw_line=True,legend_loc='lower right',scale='linear'):

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

# loop over bays, compare READ_NOISE in two BOT runs
def compare_tworuns_1d(df1,df2,run1,run2,minxy,maxxy,quantity='READ_NOISE',draw_line=True):

    # merge df's
    df_merged = pd.merge(df1, df2, how='inner', on=['BAY','SLOT','AMP'],suffixes=('_1', '_2'))

    f,ax = plt.subplots(1,1,figsize=(7,7),constrained_layout=True)
    axf = ax

    axf.hist(df_merged[quantity+'_1']/df_merged[quantity+'_2'],bins=100,range=(minxy,maxxy))
    axf.set_xlabel('%s Run %s / Run %s' % (quantity,run1,run2))

# loop over bays, compare READ_NOISE in two BOT runs
def difference_tworuns_1d(df1,df2,run1,run2,minxy,maxxy,quantity='READ_NOISE',draw_line=True):

    # merge df's
    df_merged = pd.merge(df1, df2, how='inner', on=['BAY','SLOT','AMP'],suffixes=('_1', '_2'))

    f,ax = plt.subplots(1,1,figsize=(7,7),constrained_layout=True)
    axf = ax

    axf.hist(df_merged[quantity+'_1'] - df_merged[quantity+'_2'],bins=100,range=(minxy,maxxy))
    axf.set_xlabel('%s Run %s -  Run %s' % (quantity,run1,run2))


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)    