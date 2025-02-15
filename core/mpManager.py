import ctypes as _ctypes
import multiprocessing as _mp
import numpy as _np
import psutil as _psutil
import queue as _queue
import threading as _threading
import time as _time


# Manager process to kill processes when completed or host is stopped
def processManager(num_threads, target, args, active, print_queue):
    host_proc = _psutil.Process().parent()
    
    processes = []
    for thread_id in range(num_threads):
        process = _mp.Process(target=_mpWrapper, args=(thread_id, target, args, print_queue))
        process.start()
        processes.append(process)
        if not (active.value and host_proc.is_running()):
            break

    while (active.value and host_proc.is_running()):
        any_proc_active = False
        for proc in processes:
            if proc.is_alive():
                any_proc_active = True
                break
        if not any_proc_active:
            with active.get_lock():
                active.value = False
            break
        
        _time.sleep(0.1)

    for proc in processes:
        if proc.is_alive():
            proc.terminate()
        

# Wrapper for setup and error handling
def _mpWrapper(thread_id, target, args, print_queue):
    globals()["print_queue"] = print_queue
    _ = _np.seterr(over="ignore", under="ignore")
    
    try:
        target(thread_id, *args)
    except Exception as e:
        err_msg = repr(e)
        printMP(f"Thread {thread_id} exception: {err_msg}")


# Send args to host process to print
printMP = lambda *args, sep=" ", end="\n": print_queue.put((args, sep, end))


# Enumerate new queue values
def enumQueue(q):
    while True:
        try:
            yield q.get_nowait()
        except _queue.Empty:
            break

# Clear queue
def clearQueue(q):
    while not q.empty():
        try:
            _ = q.get_nowait()
        except _queue.Empty:
            break


# Wait for barrier to pass
def barrierSync(b, timeout=1):
    try:
        b.wait(timeout)
    except _threading.BrokenBarrierError:
        raise RuntimeError("Failed to sync processes")
    b.reset()


# Managed multiprocesses
class ProcessGroup():
    def __init__(self, num_threads, target, args=tuple(), exit_timeout=1):
        self.num_threads = num_threads
        self.target = target
        self.args = args
        self.exit_timeout = exit_timeout

    def __enter__(self):
        self.__manager_active = _mp.Value(_ctypes.c_bool, True)
        self.__print_queue = _mp.Queue(maxsize=self.num_threads*10)
        
        print("\nCreating multiprocesses\n")
        self.__manager = _mp.Process(target=processManager, args=(self.num_threads, self.target, self.args, self.__manager_active, self.__print_queue))
        self.__manager.start()
        
        return self

    def __exit__(self, *args):
        with self.__manager_active.get_lock():
            self.__manager_active.value = False
            
        self.__manager.join(self.exit_timeout)
        if self.__manager.is_alive():
            raise RuntimeError("Manager failed to stop multiprocesses")

        print("\nStopped multiprocesses\n")

    def print(self):
        for args, sep, end in enumQueue(self.__print_queue):
            print(*args, sep=sep, end=end)

    clearPrintQueue = lambda self: clearQueue(self.__print_queue)

    active = property(fget = lambda self: self.__manager_active.value)
