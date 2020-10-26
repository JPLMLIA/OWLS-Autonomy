# Real Time Memory Event Plotter

This tool can easily track the memory usage processes in a script. Developed initially as a hack to watch some ORCHIDS scripts having memory issues, I cleaned this up some to be usable in external projects.

There are two pieces to this module that will be wanted and both come from `plotter.py`: `Plotter` and `watcher`.

`Plotter` is the plotting class that handles all of the actual plotting of the data as well as manage setting up the queues needed to plot. I recommend leaving the defaults as is. This simply needs to be imported, instantiated to a variable, then call the `start()` function before the rest of the script and `stop()` at the end.

`watcher` is the function that starts up the memory tracking thread and feeds data to the `Plotter` queues to graph in real time. The PID of all processes need to be known prior to starting the thread which can be difficult to implement into multiprocessing scripts as is.

## Setup

Take a look at `test_plotter.py` for an example of it working. You must import both `Plotter` and `watcher` to set up the plotter and memory tracker thread respectively.

First, instantiate the plotter with the desired refresh rate and history to be kept. The default refresh rate is 1 second and history set to 60 seconds. It is not recommended to go below 0.1 second refresh as it can become very laggy.

Next you have to set up the multiprocessing queues to track memory usage and events. The events queue isn't required, but it's helpful to mark milestones in a function on the graph. Any push to the events queue will attach that as an annotation on the graph to the closest point for its given line.

```python
readQ = pltt.get_queues('reader')
writQ = pltt.get_queues('writer')
...
readQ.event.put('Finished reading')
...
writQ.event.put('Finished writing')
```

Currently does not support adding queues after `watcher()` has been called, so they have to be instantiated all at once in the beginning then passed to other functions/processes or globalized.

A short example:

```python
import os
from plotter import Plotter, watcher

def some_func(queues, *args, **kwargs):
  do something
  ...
  some big processing here, see a spike in memory usage
  queues.event.put('Reason for spike')
  ...
  some other stuff
  ...
  finished
  queues.event.put('Finished')

if __name__ == '__main__':
  # setup the plotter
  pltt = Plotter()
  globalQ = pltt.get_queues('script_name')

  # Set up the watcher arguments
  watch = {
    'script_name': {'queue': globalQ.graph, 'pid': os.getpid()}
  }
  # Start watcher then the plotter
  watcher(watch)
  pltt.start()

  # Run the script
  some_func(globalQ)
  ...

  # Clean up at the end
  pltt.stop()
```

### Notes

The way the plotter works is that it creates its own process alongside the script processes. Due to this and the fact that it's to be in real time means that it is always listening at the refresh rate. Occasionally plotter can either get ahead of the graph queues or fall behind, so the plot might have a gap (get ahead) or skip a data point (fall behind). This occurs rather rarely from my testing, though.

Plotter is not the most efficient implementation. If the history is too long or the refresh rate is too quick the plot can experience significant lag in drawing. This is because the plot needs to be redrawn at every refresh. There is the alternative `blitter.py` that was an attempt at improving plotter, but it's broken and had its own set of unique problems. Ideally the architecture should be overhauled if this is to be considered as a serious tool for projects going forward.

Queues have to be set up before the watcher function is called or plotter is started. This is because watcher only watches what is passed to it and plotter instantiates the queues before entering its own process (at start). Attempting to add queues after starting plotter will result in those queues not correctly being added to the refresh list of plotter. Again, a refactor is needed to fix this.

Watcher requires the queue pairs for each process as well as the PID of each process to be known prior to starting watcher. This means that only processes that start at the very beginning of the script can be tracked for their memory usage. In single process scripts this okay as the PID is known at run time. In multiprocessing scripts, the only case where this module will be helpful is when the processes are spawned before watcher and plotter start.

The module isn't installable right now (so it's not a _real_ module). Packages that need to be installed for this to work are:
- psutil
- matplotlib

The test script also requires:
- numpy
