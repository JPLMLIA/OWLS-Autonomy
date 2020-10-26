#%%
#%load_ext autoreload
#%autoreload 2
import multiprocessing as mp
import numpy as np
import os
import sys
import pytest
from time  import sleep
from types import SimpleNamespace as SN

from utils.memory_tracker.blitter import Blitter
#%%

refresh = 3
history = 12
dataEvents = 1000

history / refresh

def reader(queues):
    for val in np.random.rand(dataEvents):
        queues.graph.put(val)
        if round(np.random.rand()-.25) == 1:
            queues.event.put('R')
        sleep(refresh)

def writer(queues):
    for val in np.random.rand(dataEvents):
        queues.graph.put(val)
        if round(np.random.rand()-.25) == 1:
            queues.event.put('W')
        sleep(refresh)

@pytest.mark.skip
def test_plotter():
    pltt = Blitter(refresh=refresh, history=history)

    print('Getting queues')
    readQ  = pltt.get_queues('reader')
    writeQ = pltt.get_queues('writer')

    print('Starting')
    #plttProc = mp.Process(target=pltt.start)
    readProc = mp.Process(target=reader, args=(readQ, ))
    writProc = mp.Process(target=writer, args=(writeQ, ))

    #plttProc.start()
    readProc.start()
    writProc.start()
    pltt.start()

    #sleep(5)
    #print('Pausing')
    #pltt.pause()
    #sleep(10)
    #print('Resuming')
    #pltt.resume()
    #sleep(5)

    print('Joining')
    readProc.join()
    writProc.join()
    #pltt.pause()
    #input('Press any key to exit\n')
    pltt.stop()
    #plttProc.terminate()
    print('Exiting')

@pytest.mark.skip
def test_watcher():
    pltt = Blitter(refresh=refresh, history=history)

    print('Getting queues')
    readQ = pltt.get_queues('reader')
    writQ = pltt.get_queues('writer')
    totaQ = pltt.get_queues('total')
    trash = SN(graph=mp.Queue(), event=mp.Queue())

    readProc = mp.Process(target=reader, args=(trash, ))
    writProc = mp.Process(target=writer, args=(trash, ))

    print('Starting reader and writer')
    readProc.start()
    writProc.start()

    print('Starting watcher')
    procs = {
        'reader': {'pid': readProc.pid, 'queue': readQ.graph},
        'writer': {'pid': writProc.pid, 'queue': writQ.graph},
        'total' : {'queue': totaQ.graph}
    }
    watcher(procs, refresh=refresh)

    print('Starting Blitter')
    pltt.start()

    print('Joining reader and writer')
    readProc.join()
    writProc.join()

    #pltt.pause()
    #input('Press any key to exit')
    pltt.stop()
    print('Exiting')

@pytest.mark.skip
#@watch(name='script', plotter={'refresh': refresh, 'history': history})
def test_decorator():
    data = np.array()
    for i in range(dataEvents):
        data = np.concat(data, np.random.rand(int(np.random.rand()*1e5)))
        if round(np.random.rand()-.5) == 1:
            del(data)
            data = np.array()

