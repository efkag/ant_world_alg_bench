from source import alg_bench
from multiprocessing import Process


def main():
    results_path = '../Results/bench-results-pm-newgrid.csv'

    parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
                  'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}

    routes = [1, 2]
    alg = 'spm'
    bench = alg_bench.Benchmark(results_path)
    bench.benchmark(parameters, routes)

    # bench.bench_paral(parameters, routes)


if __name__ == "__main__":
    main()
