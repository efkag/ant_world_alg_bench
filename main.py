import alg_bench

def main():
    window_range = list(range(8, 10))
    routes = [1]
    bench = alg_bench.benchmark()
    bench.benchmark_init(routes, window_range)


if __name__ == "__main__":
    main()
