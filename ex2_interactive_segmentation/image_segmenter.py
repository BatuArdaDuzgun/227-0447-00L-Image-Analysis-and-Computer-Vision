#from matplotlib.colors import makeMappingArray
import numpy as np
import random
import math


# Import maxflow (PyMaxflow library) which can be used to solve min-cut problem
import maxflow

# Set seeds for random generators to get reproducible results
random.seed(0)
np.random.seed(0)

def perform_min_cut(unary_potential_foreground, unary_potential_background, pairwise_potential):
    """
    We provide a simple fuction to perform min cut using PyMaxFlow library. You
    may use this function to implement your algorithm if you wish. Feel free to
    modify this function as desired, or implement your own function to perform
    min cut.  

    args:
        unary_potential_foreground - A single channel NumPy array specifying the
            source (foreground) unary potentials for each pixel in the image
        unary_potential_background - A single channel NumPy array specifying the
            sink (background) unary potentials for each pixel in the image
        pairwise_potential - A single channel NumPy array specifying the pairwise
            potentials. We assume a graph where each pixel in the image is 
            connected to its four neighbors (left, right, top, and bottom). 
            Furthermore, we assume that the pairwise potential for all these 4
            edges are same, and set to the value of pairwise_potential at that 
            pixel location
    """    
    
    # create graph
    maxflow_graph = maxflow.Graph[float]()
    
    # add a node for each pixel in the image
    nodeids = maxflow_graph.add_grid_nodes(unary_potential_foreground.shape[:2])

    # Add edges for pairwise potentials. We use 4 connectivety, i.e. each pixel 
    # is connected to its 4 neighbors (up, down, left, right). Also we assume
    # that pairwise potential for all these 4 edges are same
    # Feel free to change this if you wish
    maxflow_graph.add_grid_edges(nodeids, pairwise_potential)

    # Add unary potentials
    maxflow_graph.add_grid_tedges(nodeids, unary_potential_foreground, unary_potential_background)

    maxflow_graph.maxflow()
    
    # Get the segments of the nodes in the grid.
    mask_bg = maxflow_graph.get_grid_segments(nodeids)
    mask_fg = (1 - mask_bg.astype(np.uint8))* 255

    return mask_fg


class ImageSegmenter:
    def __init__(self):
        pass
    
    def segment_image(self, im_rgb, im_aux, im_box):
        # TODO: Modify this function to implement your algorithm

        """
        # Perform simple min cut
        box = im_box.astype(np.float32) / 255
        
        # creating a 3D mask
        box_mask = np.zeros(np.shape(im_rgb))

        box_mask[:,:,0] = box
        box_mask[:,:,1] = box
        box_mask[:,:,2] = box

        #test = np.multiply(im_rgb/ 255, box_mask) # np.ma.masked_array(im_rgb, mask=box_mask)
        #show_image(test, "box_mask")

        # trating array of pixel for the un-selected area
        im_selection_masked = np.ma.masked_array(im_aux, mask=box_mask)
        im_val_unselected = im_selection_masked.compressed()
        im_val_unselected = np.reshape(im_val_unselected, (-1, 3))


        # trating array of pixel for the selected area
        box_mask = np.ones(np.shape(box_mask)) - box_mask
        #test = np.multiply(im_rgb/ 255, box_mask) # np.ma.masked_array(im_rgb, mask=box_mask)
        #show_image(test, "box_mask")

        im_unselection_masked = np.ma.masked_array(im_aux, mask=box_mask)
        im_val_selected = im_unselection_masked.compressed()
        im_val_selected = np.reshape(im_val_selected, (-1, 3))




        
        # comptuting mean and variance of pixels in the two reigons) 
        selected_mean = np.mean(im_val_selected, axis=0)
        selected_cov = np.cov(im_val_selected.T)
        
        unselected_mean = np.mean(im_val_unselected, axis=0)
        unselected_cov = np.cov((im_val_unselected-unselected_mean).T)


        print("selected mean: " + str(selected_mean))
        print("selected cov: " + str(selected_cov))
        print("unselected mean: " + str(unselected_mean))
        print("unselected cov: " + str(unselected_cov))

        # calculate prob selected
        z = np.reshape(im_aux, (-1, 3))
        mean = selected_mean
        cov = selected_cov

        N = 3

        temp1 = np.linalg.det(cov) ** (-1/2)

        de_mean_z = z - mean
        inv_cov = np.linalg.inv(cov)


        temp2 = np.empty(np.shape(z)[0])
        for i in range(np.shape(temp2)[0]):

            temp2[i] = np.exp(-.5 * np.dot(np.dot((de_mean_z[i,:]), inv_cov) , (de_mean_z[i,:])))


        prob_selected = (2 * np.pi) ** (-N/2) * temp1 * temp2

        #test = np.reshape(prob_selected, np.shape(im_rgb)[:2])
        #show_image(test,"prob_selected")


        # calculate prob unselected
        z = np.reshape(im_aux, (-1, 3))
        mean = unselected_mean
        cov = unselected_cov

        N = 3

        temp1 = np.linalg.det(cov) ** (-1/2)

        de_mean_z = z - mean
        inv_cov = np.linalg.inv(cov)


        temp2 = np.empty(np.shape(z)[0])
        for i in range(np.shape(temp2)[0]):

            temp2[i] = np.exp(-.5 * np.dot(np.dot((de_mean_z[i,:]), inv_cov) , (de_mean_z[i,:])))


        prob_unselected = (2 * np.pi) ** (-N/2) * temp1 * temp2

        """

        # Perform simple min cut
        box = im_box.astype(np.float32) / 255

        # creating a 3D mask
        box_mask = np.zeros(np.shape(im_rgb))

        box_mask[:,:,0] = box
        box_mask[:,:,1] = box
        box_mask[:,:,2] = box

        #test = np.multiply(im_rgb/ 255, box_mask) # np.ma.masked_array(im_rgb, mask=box_mask)
        #show_image(test, "box_mask")

        im_aux_low_res = np.floor_divide(im_aux, 64) * 64
        #show_image(im_aux_low_res, "im_aux_low_res")


        im_selection_masked = np.ma.masked_array(im_aux_low_res, mask=box_mask)
        im_val_unselected = im_selection_masked.compressed()
        #im_val_unselected = np.reshape(im_val_unselected, (-1, 3))


        # trating array of pixel for the selected area
        box_mask = np.ones(np.shape(box_mask)) - box_mask
        #test = np.multiply(im_rgb/ 255, box_mask) # np.ma.masked_array(im_rgb, mask=box_mask)
        #show_image(test, "box_mask")


        im_unselection_masked = np.ma.masked_array(im_aux_low_res, mask=box_mask)
        im_val_selected = im_unselection_masked.compressed()
        #im_val_selected = np.reshape(im_val_selected, (-1, 3))




        im_val_selected_vec = np.reshape(im_val_selected, (-1, 3))
        im_val_unselected_vec = np.reshape(im_val_unselected, (-1, 3))

        im_val_all = np.reshape(im_aux_low_res, (-1, 3))


        selected_uniq, selected_count = np.unique(im_val_selected_vec, axis=0, return_counts=True)
        unselected_uniq, unselected_count = np.unique(im_val_unselected_vec, axis=0, return_counts=True)

        all_uniq, all_count = np.unique(im_val_all, axis=0, return_counts=True)



        selected_prob = selected_count / np.sum(selected_count)
        unselected_prob = unselected_count / np.sum(unselected_count)

        unselected_prob_new = np.empty(np.shape(selected_prob))

        for i in range(np.shape(selected_prob)[0]):

            for j in range(np.shape(unselected_prob)[0]):

                if np.array_equal(selected_uniq[i, :], unselected_uniq[j, :]):

                    unselected_prob_new[i] = unselected_prob[j]

        unselected_prob = unselected_prob_new          


        im_prob_selected = np.ones(np.shape(im_aux_low_res)[:2]) * -math.log(1e-50)
        im_prob_unselected = np.ones(np.shape(im_aux_low_res)[:2]) * -math.log(1 - 1e-50)

        
        selected_prob = -np.log(selected_prob)
        unselected_prob = -np.log(unselected_prob)


        for i in range(np.shape(im_prob_selected)[0]):

            for j in range(np.shape(im_prob_selected)[1]):

                if box[i,j]:              

                    for k in range(np.shape(selected_uniq)[0]):

                        if np.array_equal(im_aux_low_res[i, j, :], selected_uniq[k, :]):

                            im_prob_selected[i, j] = selected_prob[k]

                    #for k in range(np.shape(unselected_uniq)[0]):

                        #if np.array_equal(im_aux_low_res[i, j, :], unselected_uniq[k, :]):

                            im_prob_unselected[i, j] = unselected_prob[k]

        """
        test = np.reshape(im_prob_selected_log, np.shape(im_rgb)[:2])
        show_image(test,"prob_selected")
        test = np.reshape(im_prob_unselected_log, np.shape(im_rgb)[:2])
        show_image(test,"prob_unselected")

        test = np.reshape(im_prob_unselected_log - im_prob_selected_log, np.shape(im_rgb)[:2])
        show_image(test,"prob_unselected - prob_selected")
        """

        # Foreground potential set to 1 inside box, 0 otherwise
        unary_potential_foreground = np.reshape(im_prob_unselected, np.shape(im_box))

        # Background potential set to 0 inside box, 1 everywhere else
        unary_potential_background = np.reshape(im_prob_selected, np.shape(im_box)) 

        # Pairwise potential set to 1 everywhere
        pairwise_potential = np.ones_like(unary_potential_foreground) * 1

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_potential_foreground, unary_potential_background, 
                                  pairwise_potential)
        
        return im_mask
