#from source import cbench
from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")
from source import navbench


def static_bench():
    #results_path = f'/its/home/sk526/ant_world_alg_bench/Results/ftl/{string_date}'
    results_path = f'/mnt/data0/sk526/Results/ftl/{string_date}'
    
    #routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
    routes_path = '/mnt/data0/sk526/sussex-ftl-dataset/repeating-routes'
    # grid_path = '/home/efkag/PycharmProjects/ant_world_alg_bench/new-antworld/grid70'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True], 
                  'shape': [(360, 80), (180, 40), (90, 20)],
                  'vcrop':[0., .4, .6],
                  'histeq':[True, False],
                  'edge_range': [(180, 200), False],
                  #'loc_norm': [{'kernel_shape':(3, 3)}, False],
                  'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':[(-180, 180)],
                  'window': [0], 
                  'matcher': ['mae', 'corr'],
                  'ref_route': [1, 2, 3, 4, 5]
                  }
    
    routes = [1, 2, 3]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=None, 
                               filename='results.csv',
                               route_path_suffix='N-',
                               route_repeats=5,
                               bench_data='ftl'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=45)



def static_bench_antworld():
    #results_path = f'/its/home/sk526/ant_world_alg_bench/Results/newant/{string_date}'
    results_path = f'/mnt/data0/sk526/Results/aw/{string_date}'

    #routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
    routes_path = '/mnt/data0/sk526/new-antworld/curve-bins'

    #grid_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/grid70'
    grid_path = '/mnt/data0/sk526/new-antworld/grid70'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True], 
                  'shape': [(360, 80)],
                  #'shape':[(90, 20)],
                  #'vcrop':[0., .4, .6],
                  'histeq':[True, False],
                  'edge_range': [(180, 200), False],
                  #'loc_norm': [True, False],
                  #'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':[(-180, 180)],
                  'window': [0], 
                  'matcher': ['entropy'],
                  'grid_dist':[0.2]
                  }
    
    routes = [*range(2)]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=grid_path, grid_dist=0.2,
                               filename='results.csv',
                               bench_data='aw2'
                               )
    bench.benchmark(parameters, routes, parallel=True, cores=50)



# def live_bench():
#     #'segment_length':[3],
#     results_path = f'/its/home/sk526/ant_world_alg_bench/Results/newant/{string_date}_infomax'
#     routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
#     parameters = {'repos_thresh':[.3], 
#                   'r': [0.05], 
#                   't': [300], 
#                   'blur': [True],
#                   'shape': [(180, 80)],
#                 #  'wave' : [True, False], 
#                 #  'edge_range': [(180, 200), False],
#                 #  'loc_norm': [{'kernel_shape':(5, 5)}, False],
#                   # 'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
#                   'window': [None],
#                   'matcher': [None],
#                   }

#     routes = [*range(20)]
#     num_of_repeats = 3
#     parameters['repeat'] = [*range(num_of_repeats)]
#     cbench.benchmark(results_path, routes_path, parameters, routes, 
#                     parallel=True, num_of_repeats=num_of_repeats)


def main():
    #static_bench()
    static_bench_antworld()
    #live_bench()

if __name__ == "__main__":
    main()
