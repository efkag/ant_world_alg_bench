from source import rotate, cor_coef, pre_process  # , r_cor_coef, ridf
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import seaborn as sns

# error_path = 'loop_routes/pm/route_1_alt__error/error_1_72deg/'

# error_path = 'loop_routes/pm/route_1_alt__error/error_2_173deg/'
# grid_img_path = error_path + 'grid_35.png'
# route_img_path = error_path + 'matched_733.png'

error_path = 'Figures/spm/route_1_alt__error/rmse/error_5_95deg/'
grid_img_path = error_path + 'grid_95.png'
route_img_path = error_path + 'window_455.png'

# pre_proc = {'shape': (360, 75), 'edge_range': (180, 200)}
pre_proc = {'shape': (360, 75), 'blur': True}
grid_img = cv.imread(grid_img_path, cv.IMREAD_GRAYSCALE)
route_img = cv.imread(route_img_path, cv.IMREAD_GRAYSCALE)



def ridf(ref_img, current_img,  degrees, step):
    # degrees = round(degrees/2)
    rmse = []   # Hold the RMSEs between the current and the image of the route for every degree
    for k in range(0, degrees, step):
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
    # degrees = round(degrees/2)  # degrees to rotate for left and right
    r_coef = []   # Hold the r_coefs between the current and the image of the route for every degree
    for k in range(0, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        # coe_coef function to find the correlation between the selected route image and the rotated current
        r_coef.append(cor_coef(curr_image, ref_img))
    return r_coef


def residual_image(ref_img, current_img, d):
    current_img = rotate(d, current_img)
    res_img = np.absolute(ref_img - current_img)
    fig = plt.figure(figsize=(15, 5))
    ax = sns.heatmap(res_img)
    # ax.figure.savefig('idf_v_correlation_figures/residual image ' + str(d) + '.png')
    # save_image('idf_v_correlation_figures/residual image ' + str(d) + '.png', res_img)


def mean_residual_image(ref_img, current_img, d):
    current_img = rotate(d, current_img)
    current_img = current_img.flatten()
    ref_img = ref_img.flatten()
    current_mean = np.mean(current_img)
    ref_mean = np.mean(ref_img)
    res_img = ((ref_img - ref_mean) * (current_img - current_mean))/len(ref_img)
    res_img = np.reshape(res_img, (75, 360))
    fig = plt.figure(figsize=(15, 5))
    ax = sns.heatmap(res_img)
    # ax.figure.savefig('idf_v_correlation_figures/mean residual image ' + str(d) + '.png')
    # save_image('idf_v_correlation_figures/mean residual image ' + str(d) + '.png', res_img)



route_img = pre_process([route_img], pre_proc)[0]
grid_img = pre_process([grid_img], pre_proc)[0]


logs = ridf(route_img, grid_img, 360, 1)
print(logs.index(min(logs)))
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('RIDF')
plt.xlabel('Degrees')
plt.ylabel('IDF')
plt.show()
# Save residual images
residual_image(route_img, grid_img, 134)
residual_image(route_img, grid_img, 190)

logs = r_cor_coef(route_img, grid_img, 360, 1)
print(logs.index(max(logs)))
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('Rotational Correlation Coefficient')
plt.xlabel('Degrees')
plt.ylabel('rCorrCoeff')
plt.show()

mean_residual_image(route_img, grid_img, 190)
