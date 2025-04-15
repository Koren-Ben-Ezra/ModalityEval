import os
import threading
from queue import Queue

from src.log import Log

class JobHandler:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.job_queue = Queue()
        self.logger = Log().logger  # Cache logger once
        self.logger.info("JobHandler initialized.")
        
    def register_job(self, job):
        if not callable(job):
            raise ValueError("Job must be a callable function or method.")
        
        self.job_queue.put(job)    
        self.logger.info("Job registered.")
        
    def worker(self):
        while True:
            try:
                job = self.job_queue.get_nowait()
            except Exception:  # Usually queue.Empty
                break  # No more jobs in the queue
            
            try:
                self.logger.info("Executing job.")
                job()  # Actually run the job
            except Exception as e:
                self.logger.error(f"Error executing job: {e}")
            finally:
                self.job_queue.task_done()
                self.logger.info("Job completed.")
    
    def start_workers(self):
        # Number of worker threads to spawn
        num_threads = int(os.getenv("NUM_PARALLEL_JOBS", "1"))
        self.logger.info(f"Starting {num_threads} workers.")

        threads: list[threading.Thread] = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            threads.append(t)

        # Wait until all threads finish
        for t in threads:
            t.join()

        self.logger.info("All jobs completed.")
