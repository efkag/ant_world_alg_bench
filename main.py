import alg_bench

def main():
    results_path = 'Results/bench-results-test.csv'
    pre_processing = [dict({'blur': True, 'shape': (180, 50)})]

    # pre_processing = [{'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)},
    #                   {'blur': True, 'shape': (180, 50)},
    #                   {'blur': True, 'shape': (90, 25)}]

    # window_range = list(range(5, 20))
    window_range = list(range(6, 21))
    # routes = [1, 2, 3]
    routes = [1, 2, 3, 4]
    # matchers = ['corr', 'idf']
    matchers = ['idf']

    bench = alg_bench.Benchmark(results_path)
    bench.benchmark_init(routes, pre_processing, window_range, matchers)


if __name__ == "__main__":
    main()
