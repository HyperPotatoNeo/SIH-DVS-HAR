import struct
import os
import numpy as np
import cv2
import pandas as pd


V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def loadaerdat(datafile='/tmp/aerout.dat', length=0, version=V3, debug=1, camera='DVS128'):
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    print ("file size", length)

    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        if debug >= 2:
            print (str(lt))
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    print (xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS

        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift


            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    return timestamps, xaddr, yaddr, pol

def find_nearest(array, value):
    """Find nearest value in an array.
    Parameters
    ----------
    array : numpy.ndarray
        1-d array
    value : int
        the given value
    """
    return (np.abs(array-value)).argmin()

def find_nearest_list(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

directory='DvsGesture'

global_count=0

def cal_event_freq(event_arr, window=1000):
    """Calculate event frequence by given time window.
    Parameters
    ----------
    event_arr : numpy.ndarray
        array has 2 rows, first row contains timestamps,
        second row consists of corresponding event count at particular
        timestep
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms
    Returns
    -------
    event_freq : numpy.ndarray
        Event frequency count in given window
    """
    window=1000
    idx = 0
    tot_idx = event_arr.shape[1]-1
    event_freq = []
    while idx < tot_idx:
        #print(idx+window)
        #print(idx,min(idx+window, tot_idx))
        end_idx = find_nearest(event_arr[0, idx:min(idx+window, tot_idx)],
                               event_arr[0, idx]+window)
        end_idx = end_idx+idx+1

        event_freq.append(np.array([event_arr[0, end_idx],
                                    np.sum(event_arr[1, idx:end_idx])]))
        idx = end_idx+1

    return np.array(event_freq)

def cal_event_count(timestamps):
    """Calculate event count based on timestamps.
    Parameters
    ----------
    timestamps : numpy.ndarray
        timestamps array in 1D array
    Returns
    -------
    event_arr : numpy.ndarray
        array has 2 rows, first row contains timestamps,
        second row consists of corresponding event count at particular
        timestep
    """
    event_ts, event_count = np.unique(timestamps, return_counts=True)

    return np.asarray((event_ts, event_count))


def cal_running_std(event_freq, n=16):
    """Calculate running standard deviation.
    Parameters
    ----------
    event_freq : numpy.ndarray
        Event frequency count in given window
    n : int
        Running window for computing STD
    Returns
    -------
    o : numpy.ndarray
        Running standard deviation of given event frequency array
    """
    q = event_freq[:, 1]**2
    q = np.convolve(q, np.ones((n, )), mode="valid")
    s = np.convolve(event_freq[:, 1], np.ones((n, )), mode="valid")
    o = (q-s**2/n)/float(n-1)

    return o

def cal_first_response(timestamps, window=1000):
    """Calculate the first event burst based on standard deviation.
    Parameters
    ----------
    timestamps : numpy.ndarray
        time stamps record
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms
    Returns
    -------
    key_idx : int
        The start index of the first event burst
    """
    # Calculate event count
    event_info = cal_event_count(timestamps)

    # calculate events number within peroid of agiven window
    event_freq = cal_event_freq(event_info, window=window)

    # calculate running standard deviation
    n = 3
    o = cal_running_std(event_freq, n)

    start_idx = 0
    print(o.shape)
    while (start_idx+1<o.shape[0] and o[start_idx+1]/o[start_idx] < 3):
        start_idx += 1

    key_ts = event_freq[start_idx+2*n, 0]

    return np.where(timestamps == key_ts)[0][0]


def clean_up_events(timestamps, xaddr, yaddr, pol, window=1000,
                    key_idx=0):
    """Clean up event series based on standard deviation.
    Parameters
    ----------
    timestamps : numpy.ndarray
        time stamps record
    xaddr : numpy.ndarray
        x position of event recordings
    yaddr : numpy.ndarry
        y position of event recordings
    pol : nujmpy.ndarray
        polarity of event recordings
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms
    key_idx : int
        the timestamp that indicates the first event burst
    Returns
    -------
    Cleaned signal
    """
    if key_idx == 0:
        key_idx = cal_first_response(timestamps,window)

    return (timestamps[key_idx:], xaddr[key_idx:], yaddr[key_idx:],
            pol[key_idx:])


THRESH=0.3

for filename in os.listdir(directory):
    if(filename.endswith('labels.csv') and filename!='gesture_mapping.csv'):
        labels=pd.read_csv(directory+'/'+filename)
        aefile=filename.replace('_labels.csv','.aedat')
        print(aefile)
        timestamps,x,y,pol=loadaerdat(datafile=directory+'/'+aefile)
        zipped = list(zip(timestamps, x, y))
        zipped.sort()
        timestamps,x,y=list(zip(*zipped))
        labels=labels.values
        for i in range(labels.shape[0]):
            action=labels[i,0]-1
            start_time=labels[i,1]
            end_time=labels[i,2]
            start_time=find_nearest_list(timestamps,start_time)
            end_time=find_nearest_list(timestamps,end_time)
            start_index=timestamps.index(start_time)
            end_index=len(timestamps)-timestamps[::-1].index(end_time)
            print(end_index-start_index)
            X=np.asarray(x[start_index:end_index+1])
            T=np.asarray(timestamps[start_index:end_index+1])
            Y=np.asarray(y[start_index:end_index+1])
            polarity=np.asarray(pol[start_index:end_index+1])

            T,X,Y,polarity=clean_up_events(T,X,Y,polarity,THRESH)
            print(T.shape,X.shape,Y.shape,polarity.shape)
            X=np.asarray(X)
            if(X.shape[0]==0):
                continue
            #X=(X>>17)&0x00001FFF
            Y=np.asarray(Y)
            iter=0
            frame=np.zeros((128,128))
            #Y=(Y>>2)&0x00001FFF
            #print(X)
            polarity=np.asarray(polarity)
            for k in range(X.shape[0]):
                #print(X[k],Y[k])
                if(frame[X[k],Y[k]]==0):
                	frame[X[k],Y[k]]=polarity[k]
                iter+=1
                if(iter%500==0):
                    print(iter)
                    frame*=255
                    cv2.imwrite('parsed_data/train/'+str(action)+'/'+str(global_count)+'.png',frame)
                    global_count+=1

            #polarity=(polarity>>1)&0x00000001
            print(X[0],Y[0],np.sum(polarity),'----')
            #print(timestamps[0:100])
            break