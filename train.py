import imghdr
import numpy as np
import cv2
import os
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.network import Network
from tensorflow.python.keras.utils import generic_utils
import matplotlib.pyplot as plt
from model import model_generator, model_discriminator
import dlib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/thesis_sophie/code/image_completion_tf2-master/shape_predictor_68_face_landmarks.dat")
landmark_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,29]

class DataGenerator(object):
    def __init__(self, root_dir, image_size, local_size):
        self.image_size = image_size
        self.local_size = local_size
        self.reset()
        self.img_file_list = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if imghdr.what(full_path) is None:
                    continue
                self.img_file_list.append(full_path)

    def __len__(self):
        return len(self.img_file_list)

    def reset(self):
        self.images = []
        self.points = []
        self.masks = []
        self.paths = []

    def flow(self, batch_size, hole_min=62, hole_max=64):
        np.random.shuffle(self.img_file_list)
        for f in self.img_file_list:
            img = cv2.imread(f)
            img = cv2.resize(img, self.image_size)[:, :, ::-1]
            self.images.append(img)
            self.paths.append(f)

            x1 = 70
            y1 = 20
            x2, y2 = np.array([x1, y1]) + np.array(self.local_size)
            self.points.append([x1, y1, x2, y2])        
            

            m = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
            m[70:110,33:95] = 1
            self.masks.append(m)

            if len(self.images) == batch_size:
                inputs = np.asarray(self.images, dtype=np.float32) / 255
                points = np.asarray(self.points, dtype=np.int32)
                masks = np.asarray(self.masks, dtype=np.float32)
                paths = self.paths
                self.reset()
                yield inputs, points, masks, paths
                
                
def example_gan(result_dir="D:/thesis_sophie/code/image_completion_tf2-master/output/testing_results", 
                data_dir="D:/thesis_sophie/Data/ffhq/dataset", 
                data_test = "D:/thesis_sophie/Data/unknown_lfw"):
    input_shape = (128, 128, 3)
    local_shape = (64, 64, 3)
    batch_size = 8
    n_epoch = 35
    tc = int(n_epoch * 0.18) 
    td = int(n_epoch * 0.02)
    alpha = 0.0004

    train_datagen = DataGenerator(data_dir, input_shape[:2], local_shape[:2])
    test_datagen = DataGenerator(data_test, input_shape[:2], local_shape[:2])

    generator = model_generator(input_shape)
    discriminator = model_discriminator(input_shape, local_shape)
    optimizer = Adadelta()

    # build model
    org_img = Input(shape=input_shape)
    mask = Input(shape=(input_shape[0], input_shape[1], 1))

    in_img = Lambda(lambda x: x[0] * (1 - x[1]),
                    output_shape=input_shape)([org_img, mask])
    imitation = generator(in_img)
    completion = Lambda(lambda x: x[0] * x[2] + x[1] * (1 - x[2]),
                        output_shape=input_shape)([imitation, org_img, mask])
    cmp_container = Network([org_img, mask], completion)
    cmp_out = cmp_container([org_img, mask])
    cmp_model = Model([org_img, mask], cmp_out)
    cmp_model.compile(loss='mse',
                      optimizer=optimizer)
    cmp_model.summary()

    in_pts = Input(shape=(4,), dtype='int32')
    d_container = Network([org_img, in_pts], discriminator([org_img, in_pts]))
    d_model = Model([org_img, in_pts], d_container([org_img, in_pts]))
    d_model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)
    d_model.summary()

    d_container.trainable = False
    all_model = Model([org_img, mask, in_pts],
                      [cmp_out, d_container([cmp_out, in_pts])])
    all_model.compile(loss=['mse', 'binary_crossentropy'],
                      loss_weights=[1.0, alpha], optimizer=optimizer)
    all_model.summary()

    for n in range(n_epoch):
        progbar = generic_utils.Progbar(len(train_datagen))
        for inputs, points, masks, paths in train_datagen.flow(batch_size):
            cmp_image = cmp_model.predict([inputs, masks])
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            g_loss = 0.0
            d_loss = 0.0
            if n < tc:
                g_loss = cmp_model.train_on_batch([inputs, masks], inputs)
            else:
                d_loss_real = d_model.train_on_batch([inputs, points], valid)
                d_loss_fake = d_model.train_on_batch([cmp_image, points], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                if n >= tc + td:
                    g_loss = all_model.train_on_batch([inputs, masks, points],
                                                      [inputs, valid])
                    g_loss = g_loss[0] + alpha * g_loss[1]
            progbar.add(inputs.shape[0], values=[("D loss", d_loss), ("G mse", g_loss)])
            
        for inputs, points, masks, paths in test_datagen.flow(batch_size):
            test_image = cmp_model.predict([inputs, masks])
            for i in range(batch_size):
                os.chdir(result_dir)
                plt.imshow(test_image[i])
                plt.axis("off")
                plt.savefig(os.path.join(result_dir,  str(paths[i])[34:]), bbox_inches="tight")
                plt.close()
                
    # save model
    #generator.save(os.path.join(result_dir, "generator.h5"))
    #discriminator.save(os.path.join(result_dir, "discriminator.h5"))
    #print("saved model at epoch", n)


def main():
    example_gan()


if __name__ == "__main__":
    main()