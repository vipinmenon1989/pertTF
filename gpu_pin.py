# gpu_pin.py
import os, atexit, fcntl
try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlShutdown
    nvmlInit(); gpu_count = nvmlDeviceGetCount(); nvmlShutdown()
except ImportError:
    gpu_count = int(os.environ.get("NUM_GPUS", "1"))

locks = []
def pick_free_gpu(lock_dir="/tmp"):
    global locks
    for idx in range(gpu_count):
        lock_path = f"{lock_dir}/gpu{idx}.lock"
        lf = open(lock_path, "w")
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"[Auto-pin] using GPU {idx}", flush=True)
            locks = [lf]
            return
        except BlockingIOError:
            lf.close()
    raise RuntimeError("No free GPU lock found!")

def _cleanup():
    for lf in locks:
        try:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()
        except:
            pass
atexit.register(_cleanup)

# pick the GPU as soon as this module is imported
pick_free_gpu()

