from source2 import cor_coef,  image_split, rotate,  load_route, plot_map
import copy

# def split_eyes_sim(route_eyes, test_eyes):
#     le_v_le = cor_coef(route_eyes[0], test_eyes[0])
#     le_v_re = cor_coef(route_eyes[0], test_eyes[1])
#     re_v_re = cor_coef(route_eyes[1], test_eyes[1])
#     re_v_le = cor_coef(route_eyes[1], test_eyes[0])
#
#     return [le_v_le, le_v_re, re_v_re, re_v_le]

def split_eyes_sim(route_eyes, test_eyes):
    le_v_le = cor_coef(route_eyes[0], test_eyes[0])
    # le_v_re = cor_coef(route_eyes[0], test_eyes[1])
    re_v_re = cor_coef(route_eyes[1], test_eyes[1])
    # re_v_le = cor_coef(route_eyes[1], test_eyes[0])

    return [le_v_le, re_v_re]

matcher = 'rmse'
pre_proc = {'blur': True, 'shape': (180, 50)}
dist = 100
overlap = 45
blind = 60

w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=1, grid_pos_limit=dist)

grid_heading = copy.deepcopy(route_heading)

plot_map(w, [x_route, y_route], [x_route, y_route], size=(15, 15),
         route_headings=route_heading, grid_headings=grid_heading, scale=40)

current_image = route_images[0]
for i in range(1, len(route_images)-1):
    le, re = image_split(current_image, overlap, blind)
    route_le, route_re = image_split(route_images[i], overlap, blind)
    sims = split_eyes_sim([route_le, route_re], [le, re])
    index = sims.index(max(sims))
    if index < 1:
        current_image = rotate(5, route_images[i+1])
        grid_heading[i] = grid_heading[i-1] + 5
    else:
        current_image = rotate(-5, route_images[i + 1])
        grid_heading[i] = grid_heading[i-1] - 5


plot_map(w, [x_route, y_route], [x_route, y_route], size=(15, 15),
         route_headings=route_heading, grid_headings=grid_heading, scale=40)
