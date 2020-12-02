from source import alg_bench
from multiprocessing import Process


def main():
    results_path = '../Results/bench-results-pm-newgrid.csv'

    parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
                  'window': list(range(10, 12)), 'matcher': ['corr', 'idf']}

    routes = [1, 2]
    alg = 'spm'
    bench = alg_bench.Benchmark(results_path)
    bench.benchmark(alg, routes, parameters)

    bench.bench_paral(parameters, routes)


# def bench_proc(process, pre_proc_set):
#     results_path = 'Results/bench-results-spm-' + process + '.csv'
#     pre_processing = [pre_proc_set]
#     window_range = list(range(6, 21))
#     routes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     matchers = ['corr', 'idf']
#     # matchers = ['idf']
#     alg = 'spm'
#     bench = alg_bench.Benchmark(results_path)
#     bench.benchmark(alg, routes, params=parameters)
#     print(process + ' is complete!')

if __name__ == "__main__":
    main()
    # p1 = Process(target=bench_proc, args=('p1', {'shape': (360, 75), 'edge_range': (180, 200)}))
    # p1.start()
    #
    # p2 = Process(target=bench_proc, args=('p2', {'shape': (180, 50), 'edge_range': (180, 200)}))
    # p2.start()
    #
    # p3 = Process(target=bench_proc, args=('p3', {'shape': (90, 25), 'edge_range': (180, 200)}))
    # p3.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()

