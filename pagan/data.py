import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import os

ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
ID_ATT = {v: k for k, v in ATT_ID.items()}


def make_celeba_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        repeat=1):
    img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
    img_paths = np.array([os.path.join(img_dir, img_names.item())])
    labels = np.genfromtxt(label_path, dtype=int, usecols=(5,6,9,10,12,13,16,21,22,23,25,27,40))
    labels = np.array([labels])


    def map_fn_(img, label):
        img = tf.image.resize(img, [load_size, load_size])
        img = tl.center_crop(img, size=crop_size)
        img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
        label = (label + 1) // 2
        return img, label

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          labels=labels,
                                          drop_remainder=False,
                                          map_fn=map_fn_,
                                          shuffle=False,
                                          repeat=repeat)

    
    len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, len_dataset


def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    idx = att_names.index(att_name)

    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[idx] == 1:
            _set(att, 0, 'Bangs')
        elif att_name == 'Bangs' and att[idx] == 1:
            _set(att, 0, 'Bald')
            _set(att, 0, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[idx] == 1:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[idx] == 1:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, 0, n)

    return att_batch
