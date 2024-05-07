import datetime
from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")
from source import navbench


def static_bench():
    #results_path = f'/its/home/sk526/ant_world_alg_bench/Results/stanmer/{string_date}'
    #results_path = f'/mnt/data0/sk526/results/stanmer/{string_date}'
    results_path = f'Results/campus/{string_date}'
    
    #routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
    #routes_path = '/mnt/data0/sk526/sussex-ftl-dataset/repeating-routes'
    #routes_path = '/its/home/sk526/navlib/data/outdoors/clean/for_bench/campus'
    routes_path = 'datasets/campus'
    #routes_path = '/mnt/data0/sk526/stanmer'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True], 
                  'shape': [(180, 45)],
                  'vcrop':[0],
                  #'histeq':[True],
                  'edge_range': [(50, 255), False],
                  #'loc_norm': [{'kernel_shape':(3, 3)}, False],
                  'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':[(-90, 90)],
                  'window': [0], 
                  'matcher': ['mae'],
                  'ref_route': [1],
                  'sample_step':[5]
                  }
    
    routes = [1]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=None, 
                               filename='results.csv',
                               route_path_suffix='r',
                               route_repeats=3,
                               bench_data='bob'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=1)



def static_bench_antworld():
    results_path = f'Results/newant/static-bench/{string_date}'
    #results_path = f'/mnt/data0/sk526/Results/aw/{string_date}'

    routes_path = 'datasets/new-antworld/curve-bins'
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
                  #'edge_range': [(100, 255), False],
                  #'loc_norm': [True, False],
                  #'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':[(-180, 180)],
                  'window': [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500], 
                  'matcher': ['mae'],
                  'grid_dist':[0.2]
                  }
    
    routes = [*range(20)]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=grid_path, grid_dist=0.2,
                               filename='results.csv',
                               bench_data='aw2'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=15)



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
                  #'edge_range': [(50, 255), False],
                  #'loc_norm': [{'kernel_shape':(5, 5)}, False],
                  #'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  }
    
    nav_params = {#'pm':{'matcher':['mae']},
                  #'smw':{'window':[300, 500], 'matcher':['mae']},
                  'asmw':{'window':[-20], 'matcher':['mae']},
                  #'s2s':{'window':[20], 'queue_size':[3], 'matcher':['mae'], 'sub_window':[3]}
    }

    routes = [*range(20)]
    num_of_repeats = 3
    parameters['repeat'] = [*range(num_of_repeats)]
    cbench.benchmark(results_path, routes_path, params=parameters, nav_params=nav_params,
                    route_ids=routes, parallel=True, num_of_repeats=num_of_repeats, cores=3)


def main():
    start_dtime = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print(start_dtime)
    #static_bench()
    static_bench_antworld()
    #live_bench()
    
    end_dtime = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    print('start: ', start_dtime, ' ends: ', end_dtime)

if __name__ == "__main__":
    main()
