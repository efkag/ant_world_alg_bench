from source.utils import load_route, rotate, rmse, cor_coef, pre_process# , r_cor_coef, ridf
import matplotlib.pyplot as plt

_, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                        route_heading, route_images = load_route(1)

def ridf(ref_img, current_img,  degrees, step):
    degrees = round(degrees/2)
    rmse = []   # Hold the RMSEs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        rmse.append(rmse(curr_image, ref_img))  #IDF function to find the error between the selected route image and the rotated current
    return rmse

def r_cor_coef(ref_img, current_img,  degrees, step):
    '''
    Calculates rotational correlation coefficients
    :param ref_img:
    :param current_img:
    :param degrees:
    :param step:
    :return:
    '''
    degrees = round(degrees/2)  # degrees to rotate for left and right
    r_coef = []   # Hold the r_coefs between the current and the image of the route for every degree
    for k in range(-degrees, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        # coe_coef function to find the correlation between the selected route image and the rotated current
        r_coef.append(cor_coef(curr_image, ref_img))
    return r_coef


pre_proc = {'blur': True, 'shape': (360, 100)}
pre_route_images = pre_process(route_images, pre_proc)
img = pre_route_images[0]

logs = ridf(img, img, 360, 1)
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('RIDF')
plt.xlabel('Degrees')
plt.ylabel('IDF')
fig.savefig('test.png')
plt.show()


logs = r_cor_coef(img, img, 360, 1)
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('Rotational Correlation Coefficient')
plt.xlabel('Degrees')
plt.ylabel('rCorrCoeff')
fig.savefig('test.png')
plt.show()
