from source.utils import pre_process, mean_degree_error, cor_coef,  image_split, rotate,  load_route, plot_map
from source import sequential_perfect_memory as spm
import random

def split_eyes_sim(route_eyes, test_eyes):
    le_v_le = cor_coef(route_eyes[0], test_eyes[0])
    le_v_re = cor_coef(route_eyes[0], test_eyes[1])
    re_v_re = cor_coef(route_eyes[1], test_eyes[1])
    re_v_le = cor_coef(route_eyes[1], test_eyes[0])

    return [le_v_le, le_v_re, re_v_re, re_v_le]

matcher = 'idf'
pre_proc = {'blur': True, 'shape': (180, 50)}
dist = 100
overlap = None

w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=1, grid_pos_limit=dist)

grid_heading = [0] * len(route_heading)

plot_map(w, [x_route, y_route], [x_route, y_route], size=(15, 15),
         route_headings=route_heading, grid_headings=grid_heading, scale=40)


grid_heading = []
rotated = []
split_route_images = []
for i, v in enumerate(route_images):
    deg = random.sample([90, 270], 1)[0]
    grid_heading.append(deg)
    img = rotate(deg, v)
    le, re = image_split(img, overlap)
    rotated.append([le, re])

    le, re = image_split(v, overlap)
    split_route_images.append([le, re])


plot_map(w, [x_route, y_route], [x_route, y_route], size=(15, 15),
         route_headings=route_heading, grid_headings=grid_heading, scale=40)


for i, v in enumerate(split_route_images):
    sims = split_eyes_sim(v, rotated[i])
    index = sims.index(max(sims))
    if index < 2:
        grid_heading[i] += 30
    else:
        grid_heading[i] -= 30

plot_map(w, [x_route, y_route], [x_route, y_route], size=(15, 15),
         route_headings=route_heading, grid_headings=grid_heading, scale=40)

sims = split_eyes_sim(split_route_images[0], rotated[0])
print(sims)