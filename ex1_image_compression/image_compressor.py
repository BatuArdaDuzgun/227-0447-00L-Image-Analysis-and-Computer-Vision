# Note: You are not allowed to import additional python packages except NumPy
import numpy as np


class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self, number_of_eigen_vectors = 25):
        # You can modify the init function to add / remove some fields
        self.mean_image = np.array([])
        self.principal_components = np.array([])

        self.number_of_eigen_vectors = number_of_eigen_vectors
        
    def get_codebook(self):
        # This function should return all information needed for compression
        # as a single numpy array
        
        # TODO: Modify this according to you algorithm
        mean_image_re = np.reshape(self.mean_image, (1, 9216, 3))
        #principal_components_re = np.reshape(self.principal_components, (self.number_of_eigen_vectors, -1))
        principal_components_re = self.principal_components

        codebook = np.concatenate((mean_image_re, principal_components_re), 0).astype("float32")
        return codebook
    
    def train(self, train_images, principal_components = [], mean_image = None):
        # Given a list of training images as input, this function should learn the 
        # codebook which will then be used for compression
        
        if len(principal_components) == 0:

            self.mean_image = np.mean(train_images, axis=0)
            #print(np.shape(self.mean_image))
            #show_image(self.mean_image.astype(np.uint8), 'Mean image')
            
            demeaned_train_images = train_images - self.mean_image
            
            demeaned_train_images_R = demeaned_train_images[:, :, :, 0]
            demeaned_train_images_G = demeaned_train_images[:, :, :, 1]
            demeaned_train_images_B = demeaned_train_images[:, :, :, 2]

            #show_image(demeaned_train_images_R[0,:,:].astype(np.uint8), 'red image')
            #show_image(demeaned_train_images_G[0,:,:].astype(np.uint8), 'green image')
            #show_image(demeaned_train_images_B[0,:,:].astype(np.uint8), 'blue image')


            vector_demeaned_train_images_R = np.reshape(demeaned_train_images_R, (np.shape(train_images)[0], -1))
            vector_demeaned_train_images_G = np.reshape(demeaned_train_images_G, (np.shape(train_images)[0], -1))
            vector_demeaned_train_images_B = np.reshape(demeaned_train_images_B, (np.shape(train_images)[0], -1))

            print("stated covariance 1")
            covariance_matrix = np.dot(vector_demeaned_train_images_R.T, vector_demeaned_train_images_R)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            eigenvectors_R = eigenvectors[:,:self.number_of_eigen_vectors]

            print("finished covariance 1")

            print("stated covariance 2")
            covariance_matrix = np.dot(vector_demeaned_train_images_G.T, vector_demeaned_train_images_G)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            eigenvectors_G = eigenvectors[:,:self.number_of_eigen_vectors]

            print("finished covariance 2")

            print("stated covariance 3")
            covariance_matrix = np.dot(vector_demeaned_train_images_B.T, vector_demeaned_train_images_B)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            eigenvectors_B = eigenvectors[:,:self.number_of_eigen_vectors]

            print("finished covariance 3")


            self.principal_components = np.stack((eigenvectors_R, eigenvectors_G, eigenvectors_B), axis = 2)

            R = self.principal_components[:,:,0].T
            G = self.principal_components[:,:,1].T
            B = self.principal_components[:,:,2].T

            self.principal_components =  np.stack((R, G, B), axis = 2).real 

        else:

            self.principal_components = principal_components[:self.number_of_eigen_vectors,:,:] * (3 ** (0.5))
            self.mean_image = mean_image




        # ******************************* TODO: Implement this ***********************************************#
        return #self.principal_components, self.mean_image

    def compress(self, test_image):
        # Given a test image, this function should return the compressed representation of the image
        # ******************************* TODO: Implement this ***********************************************#
        test_image = test_image - self.mean_image
        
        test_image_vector = np.reshape(test_image, (-1, 3))



        principal_components_vector_R = self.principal_components[:,:,0]
        principal_components_vector_G = self.principal_components[:,:,1]
        principal_components_vector_B = self.principal_components[:,:,2]


        test_image_compressed_R = np.dot(principal_components_vector_R, test_image_vector[:,0].T)
        test_image_compressed_G = np.dot(principal_components_vector_G, test_image_vector[:,1].T)
        test_image_compressed_B = np.dot(principal_components_vector_B, test_image_vector[:,2].T)

        test_image_compressed = np.stack((test_image_compressed_R, test_image_compressed_G, test_image_compressed_B), axis=1).astype("int16")


        return test_image_compressed


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructor
        self.mean_image = np.reshape(codebook[0, :], (96, 96, 3))
        self.principal_components = codebook[1:,:]

        #show_image(self.mean_image * (1/255), 'mean image used for reconstuct')

    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        # ******************************* TODO: Implement this ***********************************************#
        principal_components_vector =  np.reshape(self.principal_components, (25, -1, 3))

        principal_components_vector_R = principal_components_vector[:,:,0]
        principal_components_vector_G = principal_components_vector[:,:,1]
        principal_components_vector_B = principal_components_vector[:,:,2]

        test_image_recon_R = np.dot(test_image_compressed[:,0].T, principal_components_vector_R)
        test_image_recon_G = np.dot(test_image_compressed[:,1].T, principal_components_vector_G)
        test_image_recon_B = np.dot(test_image_compressed[:,2].T, principal_components_vector_B)

        test_image_recon = np.stack((test_image_recon_R, test_image_recon_G, test_image_recon_B), axis = 1)

        test_image_recon = np.reshape(test_image_recon, (96, 96, 3)) + self.mean_image

        test_image_recon = np.heaviside(test_image_recon - 127.5, 1) * 255.0

        return test_image_recon

