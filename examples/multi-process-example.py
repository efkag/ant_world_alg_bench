from source.utils import pre_process, mean_degree_error, load_route
from source import seqnav as spm
from multiprocessing import Process


def f1(route_id):
    matcher = 'rmse'
    pre_proc = {'blur': True, 'shape': (180, 50)}
    window = 14
    dist = 100
    w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
    route_heading, route_images = load_route(route_id=route_id, grid_pos_limit=dist)

    pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
    pre_route_images = pre_process(route_images, pre_proc)

    nav = spm.SequentialPerfectMemory(pre_route_images, matcher, window=window)
    recovered_heading, window_log = nav.navigate(pre_world_grid_imgs)

    print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))


def f2(route_id):
    matcher = 'rmse'
    pre_proc = {'blur': True, 'shape': (180, 50)}
    window = 14
    dist = 100
    for id in route_id:
        w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
        route_heading, route_images = load_route(route_id=id, grid_pos_limit=dist)

        pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
        pre_route_images = pre_process(route_images, pre_proc)

        nav = spm.SequentialPerfectMemory(pre_route_images, matcher, window=window)
        recovered_heading, window_log = nav.navigate(pre_world_grid_imgs)

        # nav = pm.PerfectMemory(pre_route_images, matcher)
        # recovered_heading, logs = nav.navigate(pre_world_grid_imgs)

        print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))


p1 = Process(target=f1, args=(1, ))
p2 = Process(target=f2, args=([1, 2], ))

p1.start()
p2.start()

