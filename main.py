from train_main import train_main
from test import test_all
import multiprocessing


def run_train(start, end):
    jobs = []
    for i in range(start, end):
        p = multiprocessing.Process(target=train_main, args=(i,))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()

if __name__ == '__main__':
    run_train(0, 5)
    run_train(5, 10) 
    test_all()
