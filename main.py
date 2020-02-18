import alg_bench

def main():
    pre_processing = [dict({'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)})]#,
                      #dict({'blur': True, 'shape': (180, 50), 'edge_range': None})]

    window_range = list(range(5, 7))
    routes = [1]
    matchers = ['corr', 'idf']

    bench = alg_bench.Benchmark()
    bench.benchmark_init(routes, pre_processing, window_range, matchers)


if __name__ == "__main__":
    main()
