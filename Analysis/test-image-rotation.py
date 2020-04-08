from utils import rotate, idf, cor_coef, pre_process# , r_cor_coef, ridf
import matplotlib.pyplot as plt
import cv2 as cv

error_path = 'Images/error_33_106deg/'
grid_img_path = error_path + 'grid_70.png'
grid_img = cv.imread(grid_img_path, cv.IMREAD_GRAYSCALE)

route_img_path = error_path + 'route_341.png'
route_img = cv.imread(route_img_path, cv.IMREAD_GRAYSCALE)


def ridf(ref_img, current_img,  degrees, step):
    # degrees = round(degrees/2)
    rmse = []   # Hold the RMSEs between the current and the image of the route for every degree
    for k in range(0, degrees, step):
        curr_image = rotate(k, current_img)    #Rotate the current image
        rmse.append(idf(curr_image, ref_img))  #IDF function to find the error between the selected route image and the rotated current
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


pre_proc = {'blur': True, 'shape': (180, 50)}
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

logs = r_cor_coef(route_img, grid_img, 360, 1)
print(logs.index(max(logs)))
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('Rotational Correlation Coefficient')
plt.xlabel('Degrees')
plt.ylabel('rCorrCoeff')
plt.show()
