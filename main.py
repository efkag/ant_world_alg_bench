import alg_bench
from multiprocessing import Process


def main():
    results_path = 'Results/bench-results-pm.csv'
    # pre_processing = [dict({'blur': True, 'shape': (180, 50)})]

    pre_processing = [{'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)},
                      {'blur': True, 'shape': (180, 50)},
                      {'blur': True, 'shape': (90, 25)}]

    # window_range = list(range(5, 20))
    window_range = list(range(6, 21))
    # routes = [1, 2, 3]
    routes = [1, 2, 3, 4, 5]
    matchers = ['corr', 'idf']
    # matchers = ['idf']
    alg = 'pm'
    bench = alg_bench.Benchmark(results_path)
    bench.benchmark_init(alg, routes, pre_processing, window_range, matchers)


def bench_proc(process, pre_proc_set):
    results_path = 'Results/bench-results-pm-' + process + '.csv'
    pre_processing = [pre_proc_set]
    window_range = list(range(6, 8))
    routes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    matchers = ['corr', 'idf']
    # matchers = ['idf']
    alg = 'pm'
    bench = alg_bench.Benchmark(results_path)
    bench.benchmark_init(alg, routes, pre_processing, window_range=window_range, matchers=matchers)
    print(process + ' is complete!')

if __name__ == "__main__":
    # main()
    # p1 = Process(target=bench_proc, args=('p1', {'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)}))
    # p1.start()

    p2 = Process(target=bench_proc, args=('p2', {'blur': True, 'shape': (180, 50)}))
    p2.start()

    p3 = Process(target=bench_proc, args=('p3', {'blur': True, 'shape': (90, 25)}))
    p3.start()

    p2.join()
    p3.join()

