import os

os.system("pkill python")

from multiprocessing.shared_memory import SharedMemory

name = 'buffer'

shm = SharedMemory(name, create=False)

shm.unlink()
