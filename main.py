import datetime
from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")
from source import navbench


def static_bench():
    #results_path = f'/its/home/sk526/ant_world_alg_bench/Results/stanmer/{string_date}'
    results_path = f'/mnt/data0/sk526/results/stanmer/{string_date}'
    #results_path = f'Results/stanmer/{string_date}'
    
    #routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
    #routes_path = '/mnt/data0/sk526/sussex-ftl-dataset/repeating-routes'
    #routes_path = '/its/home/sk526/navlib/data/outdoors/clean/stanmer'
    #routes_path = 'datasets/stanmer'
    routes_path = '/mnt/data0/sk526/stanmer'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True], 
                  'shape': [(180, 45)],
                  #'vcrop':[.5],
                  #'histeq':[True],
                  'edge_range': [(50, 255), False],
                  #'loc_norm': [{'kernel_shape':(3, 3)}, False],
                  'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':[(-90, 90)],
                  'window': [0], 
                  'matcher': ['mae'],
                  'ref_route': [1, 2, 3, 4],
                  'sample_step':[2]
                  }
    
    routes = [1]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=None, 
                               filename='results.csv',
                               route_path_suffix='r',
                               route_repeats=4,
                               bench_data='bob'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=20)



def static_bench_antworld():
    results_path = f'Results/newant/{string_date}_temp'
    #results_path = f'/mnt/data0/sk526/Results/aw/{string_date}'

    routes_path = 'datasets/new-antworld/exp1'
    #routes_path = '/mnt/data0/sk526/new-antworld/curve-bins'

    grid_path = 'datasets/new-antworld/grid70'
    #grid_path = '/mnt/data0/sk526/new-antworld/grid70'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True], 
                  'shape': [(180, 40)],
                  #'shape':[(90, 20)],
                  #'vcrop':[0., .4, .6],
                  #'histeq':[True, False],
                  'edge_range': [(190, 240), False],
                  #'loc_norm': [True, False],
                  #'gauss_loc_norm': [{'sig1':2, 'sig2':20}],
                  'deg_range':[(-180, 180)],
                  'window': [0], 
                  'matcher': ['mae', 'ccd'],
                  'grid_dist':[0.2]
                  }
    
    routes = [1, 2, 3, 4, 5, 6, 7]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=grid_path, grid_dist=0.2,
                               filename='results.csv',
                               bench_data='aw2'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=55)



def live_bench():
    from source import cbench
    #'segment_length':[3],
    results_path = f'Results/newant/{string_date}'
    routes_path = 'datasets/new-antworld/curve-bins'
    parameters = {'repos_thresh':[.3], 
                  'r': [0.05], 
                  't': [1000], 
                  'blur': [True],
                  'shape': [(180, 40)],
                  'deg_range':[(-180, 180)],
                  #'w_thresh': [0.05],
                 # 'sma_size': [5],
                #  'wave' : [True, False], 
                  #'edge_range': [(180, 200), False],
                #  'loc_norm': [{'kernel_shape':(5, 5)}, False],
                 # 'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'window': [-15, 10, 15, 20, 25, 30, 40, 50],
                  'matcher': ['mae'],
                  }

    routes = [*range(20)]
    num_of_repeats = 3
    parameters['repeat'] = [*range(num_of_repeats)]
    cbench.benchmark(results_path, routes_path, parameters, routes, 
                    parallel=True, num_of_repeats=num_of_repeats, cores=6)


def main():
    start_dtime = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print(start_dtime)
    static_bench()
    #static_bench_antworld()
    #live_bench()
    
    end_dtime = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    print('start: ', start_dtime, ' ends: ', end_dtime)

if __name__ == "__main__":
    main()
