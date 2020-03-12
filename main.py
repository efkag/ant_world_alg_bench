import alg_bench

def main():
    results_path = 'Results/bench-results-full.csv'
    # pre_processing = [dict({'blur': True, 'shape': (180, 50)})]

    pre_processing = [{'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)},
                      {'blur': True, 'shape': (180, 50)}]

    # window_range = list(range(5, 20))
    window_range = list(range(5, 13))
    # routes = [1, 2, 3]
    routes = [1, 2]
    # matchers = ['corr', 'idf']
    matchers = ['corr', 'idf']

    bench = alg_bench.Benchmark(results_path)
    bench.benchmark_init(routes, pre_processing, window_range, matchers)


if __name__ == "__main__":
    main()
