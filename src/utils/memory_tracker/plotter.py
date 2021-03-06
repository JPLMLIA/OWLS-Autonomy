#%%
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import psutil

from multiprocessing import Event,   \
                            Process, \
                            SimpleQueue as SQ, \
                            set_start_method
from threading       import Thread

from time  import sleep
from types import SimpleNamespace as SN

set_start_method('spawn', force=True)

class QueuePair(object):
    def __init__(self, name, history, padding=0):
        self.graph  = SQ()
        self.event = SQ()

        self.name    = name
        self.history = history
        self.y       = SN(graph=[None]*padding, event=[None]*padding)
        self.anno    = []

    def _get_graph(self):
        val = None
        if not self.graph.empty():
            val = self.graph.get()
        self.y.graph.append(val)
        self.y.graph = self.y.graph[-self.history:]

    def _get_event(self, t):
        val = None
        # If a message was recieved
        if not self.event.empty():
            val = self.event.get()
            # Attach it to the last known data event
            for i in range(1, len(self.y.graph)):
                if self.y.graph[-i] is not None:
                    self.anno.append([t, self.y.graph[-i], str(val)])
                    val = self.y.graph[-i]
                    break
            else:
                print(f'Could not attach annotation {val} to a value on the graph, dropping annotation')
                val = None
        self.y.event.append(val)
        self.y.event = self.y.event[-self.history:]

    def get(self, t):
        self._get_graph()
        self._get_event(t)
        return

class Plotter(object):
    def __init__(self, refresh=1, history=60, style='fivethirtyeight', save_to=None):
        self.queues  = {}
        self.refresh = refresh
        self.history = int(history / refresh)
        self.time    = 0
        self.x       = []
        #self.wait    = Event()
        self.save_wait = Event()
        self.save_wait.clear()
        self.save_to   = save_to
        # Instantiate the plot
        self.style = style
        self.fig = plt.figure()
        plt.ion()

    def get_queues(self, name):
        if name in self.queues:
            return self.queues[name]
        self.queues[name] = QueuePair(name=name, history=self.history)
        return self.queues[name]

    def start(self):
        #self.wait.set()
        #self.proc = Process(target=self.__run__)
        #self.proc.start()
        proc = Process(target=self._run)
        proc.start()
        self.proc = psutil.Process(proc.pid)
        return True

    def stop(self):
        self.save_wait.set()
        sleep(.1)
        self.save_wait.wait()
        self.proc.terminate()
        return True

    def _run(self):
        # TODO: look into blit support
        self.anim = animation.FuncAnimation(self.fig, self.animation, interval=self.refresh*1000, cache_frame_data=False)
        plt.style.use(self.style)
        if self.save_to:
            self.save_wait.wait()
            self.save_wait.clear()
            self.anim.save(self.save_to)
            self.save_wait.set()
        else:
            plt.show()

        '''
        while True:
            if not self.wait.is_set():
                print('PAUSING')
                self.anim.event_source.stop()
                self.wait.wait()
                print('RESUMING')
                self.anim.event_source.start()
            sleep(self.refresh)
        '''

    def pause(self):
        self.proc.suspend()
        #self.wait.clear()
        return
        self.anim.event_source.stop()

    def resume(self):
        self.proc.resume()
        #self.wait.set()
        return
        self.anim.event_source.start()

    def animation(self, i):
        self.time += self.refresh
        self.x.append(self.time)
        self.x = self.x[-self.history:]

        plt.clf()
        for name, pair in self.queues.items():
            pair.get(self.time)
            plt.plot(self.x, pair.y.graph, label=pair.name)
            plt.plot(self.x, pair.y.event, 'o')
            for anno in pair.anno:
                x, y, note = anno
                plt.annotate(note, xy=(x, y))

        plt.title('Memory Usage over Time')
        plt.ylabel('GB')
        plt.xlabel(f'Time ({self.refresh}s)')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.legend(loc='upper left')

"""# NOT IMPLEMENTED
def dec_watch(func, name=None, enable=[], pArgs={}, wArgs={}, queues=[]):
    '''
    queues = ['name', 'name', ...]
    return queues for those names into the target func
    '''
    def wrap(*args, **kwargs):
        return func(*args, **kwargs, pltt=ret)

    pltt = Plotter(**pArgs)
    ret = {'pltt': pltt}
    ret[name or func.__name__] = {'pid': psutil.os.getpid(), 'queues': pltt.get_queues(name or func.__name__)}

    for resv in enable:
        if resv in ['total']:
            watch[resv] = {resv: {}, pltt.get_queues(resv)}

    watcher(watch=ret, **wArgs)
    return wrap
"""

def watcher(*args, **kwargs):
    th = Thread(target=_watch, args=args, kwargs=kwargs, daemon=True)
    th.daemon = True
    th.start()

def _watch(watch, refresh=1, measure=1e9, limit=10, **kwargs):
    '''
    Procs is a dict in the format:
    {name: {pid: ..., queue: ...}, ...}
    '''
    print('Initializing watcher')

    resvd = ['total']
    # Separate reserved processes from nonreserved
    procs = [SN(name=name, **attrs) for name, attrs in watch.items() if name not in resvd]
    resvs = [SN(name=name, **attrs) for name, attrs in watch.items() if name in resvd]

    # Initialize the psutil process objects and expected variables
    for proc in procs:
        proc.ps  = psutil.Process(proc.pid)
        proc.mem = 0
        proc.max = 0
    for resv in resvs:
        resv.mem = 0
        resv.max = 0

    try:
        alive = True
        while alive:
            alive = False
            for proc in procs:
                if not proc.ps.is_running() or proc.ps.status() == psutil.STATUS_ZOMBIE:
                    proc.mem = 0
                    continue
                alive = True
                proc.mem = proc.ps.memory_info().rss / measure
                proc.max = max(proc.max, proc.mem)
                proc.queue.put(proc.mem)
            for resv in resvs:
                if resv.name == 'total':
                    resv.mem = sum([proc.mem for proc in procs])
                    resv.max = max(resv.max, resv.mem)
                    resv.queue.put(resv.mem)
            sleep(refresh)
    except:
        pass

    print('Closing watcher')

def __watch(watch, refresh=1, measure=1e9, limit=10, timeout=30):
    print('Initializing watcher')

    resvd = ['total']
    # Separate reserved processes from nonreserved
    procs = [SN(name=name, **attrs) for name, attrs in watch.items() if name not in resvd]
    resvs = [SN(name=name, **attrs) for name, attrs in watch.items() if name in resvd]

    # Initialize the psutil process objects and expected variables
    for proc in procs:
        proc.ps  = psutil.Process(proc.pid)
        proc.mem = 0
        proc.max = 0
    for resv in resvs:
        resv.mem = 0
        resv.max = 0


    health = timeout
    change = False
    while health:
        for proc in procs:
            if not proc.ps.is_running() or proc.ps.status() == psutil.STATUS_ZOMBIE:
                proc.mem = 0
                continue
            proc.prv = proc.mem
            proc.mem = proc.ps.memory_info().rss / measure
            proc.max = max(proc.max, proc.mem)
            proc.queue.put(proc.mem)
            # If the previous memory is different than the current
            if proc.prv != proc.mem:
                change = True
        for resv in resvs:
            if resv.name == 'total':
                resv.mem = sum([proc.mem for proc in procs])
                resv.max = max(resv.max, resv.mem)
                resv.queue.put(resv.mem)
        sleep(refresh)
        # If change did occurred, reset health, else decrement
        if change:
            change = False
            health = timeout
        else:
            health -= 1
