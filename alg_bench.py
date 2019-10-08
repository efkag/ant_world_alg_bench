import utils as utl

world, X, Y, world_grid_imgs, X_route, Y_route, route_heading, route_images = utl.load_route("2")

utl.plot_map(world, [X_route, Y_route], [X, Y])


