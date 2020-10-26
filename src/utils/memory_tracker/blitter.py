#%%
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import psutil

from multiprocessing import Event,   \
                            Manager, \
                            Process, \
                            SimpleQueue as SQ, \
                            set_start_method
from threading       import Thread

from time  import sleep
from types import SimpleNamespace as SN

set_start_method('spawn', force=True)
#plt.ion()

#%%

class QueuePair(object):
    def __init__(self, name, refresh, history, step, padding=0):
        self.graph = SQ()
        self.event = SQ()

        self.name    = name
        self.refresh = refresh
        self.history = history
        self.step    = step
        self.y       = SN(graph=[], event=[])

    def create_plots(self, ax):
        self.ax = ax
        self.plots = SN(
                    graph = ax.plot(np.linspace(0, self.history, self.refresh), label=self.name)[0],
                    event = ax.plot(np.linspace(0, self.history, self.refresh), 'o')[0],
                    annotations = []
                    )
        #self.plots.graph.set_xdata(range(0, history))
        #self.plots.graph.set_xdata(range(0, history))
        #self.plots.graph.set_ydata(range(0, history))

    def annotate(self, xy, note):
        self.plots.annotations.append(self.ax.annotate(note, xy=xy))

    def _update_y(self, plot, y, val):
        print(len(self.plots.graph.get_xdata()))
        y.append(val)
        if len(y) < self.step:
            print(f'Case 1: len({len(y + [np.nan]*(self.step-len(y)))})')
            plot.set_ydata(y + [np.nan]*(self.step-len(y)))
        else:
            y.pop(0)
            plot.set_ydata(y)
        #if self.t >= self.history:
            s, f = self.ax.get_xlim()
            plot.set_xdata(range(int(s+self.refresh), int(f+self.refresh)))


    def _next_graph(self):
        val = self.graph.get() if not self.graph.empty() else np.nan
        self._update_y(self.plots.graph, self.y.graph, val)
        self.plots.min = np.nanmin(self.y.graph)
        self.plots.max = np.nanmax(self.y.graph)

    def _next_event(self):
        val = np.nan
        # If a message was recieved
        if not self.event.empty():
            val = self.event.get()
            # Attach it to the last known data event
            for i in range(1, len(self.y.graph)):
                if not np.isnan(self.y.graph[-i]):
                    x = self.t+1 if self.t < self.history else self.t-1
                    y = self.y.graph[-i]
                    self.annotate(xy=(x, y), note=str(val))
                    val = self.y.graph[-i]
                    break
            else:
                print(f'Could not attach annotation {val} to a value on the graph, dropping annotation')
                val = np.nan

        self._update_y(self.plots.event, self.y.event, val)

    def _clean_anno(self):
        for anno in self.plots.annotations:
            if anno.xy[0] < self.t:
                self.plots.annotations.remove(anno)

    def next(self, t):
        self.t = t
        self._next_graph()
        self._next_event()
        #self._clean_anno()
        return self.plots.graph, self.plots.event, self.plots.annotations

class Blitter(object):
    def __init__(self, refresh=1, history=120, style='fivethirtyeight'):
        self.queues  = {}
        self.refresh = refresh
        self.history = history
        self.step    = round(history / refresh)

        self.style = style
        #self.fig, self.ax = plt.subplots()

        #self.mg = Manager()
        #self.ns = self.mg.Namespace()
        #self.ns.fig, self.ns.ax = plt.subplots()

    def get_queues(self, name):
        if name in self.queues:
            return self.queues[name]
        self.queues[name] = QueuePair(name=name, refresh=self.refresh, history=self.history, step=self.step)
        return self.queues[name]

    def start(self):
        #self.wait.set()
        #self.proc = Process(target=self.__run__)
        #self.proc.start()
        proc = Process(target=self.run)
        proc.start()
        self.proc = psutil.Process(proc.pid)
        return True

    def run(self):
        # TODO: look into blit support
        self.fig, self.ax = plt.subplots()

        self.ax.set_title('Memory Usage over Time')
        self.ax.set_ylabel('Memory (GB)')
        self.ax.set_xlabel(f'Time ({self.refresh}s)')
        self.ax.set_xlim(0, self.history)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        for name, pair in self.queues.items():
            pair.create_plots(ax=self.ax)

        self.anim = animation.FuncAnimation(self.fig, self.animation, interval=self.refresh*1000, blit=False, init_func=lambda: None)

        plt.style.use(self.style)
        plt.show()

    def stop(self):
        self.proc.terminate()
        return True

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
        t = i * self.refresh
        print(f't: {t}')
        #plots = []
        minmax = []
        for name, pair in self.queues.items():
            graph, event, annotations = pair.next(t)
            #plots.append(graph)
            #plots.append(event)
            #plots += annotations
            minmax.append(pair.plots.min)
            minmax.append(pair.plots.max)

        #if t >= self.history:
        if t > self.history:
            s, f = self.ax.get_xlim()
            self.ax.set_xlim(s+self.refresh, f+self.refresh)

        self.ax.set_ylim(np.nan_to_num(np.nanmin(minmax)), np.nan_to_num(np.nanmax(minmax)) or 1)
        self.ax.legend(loc='upper left')
        self.ax.figure.canvas.draw()

        return []


#%%
