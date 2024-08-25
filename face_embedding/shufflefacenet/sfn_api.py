# -- coding: utf-8 --
# @Time : 2024/8/25
# @Author : ykk648

from src.shufflefacenet import Network
import torch
import cv2


def init_model():
    model_path = 'private_models/face_lib/face_embedding/ShuffleFaceNet-1.5-d512.pth'
    model = Network(num_classes=512, width_multiplier=1.5, input_size=(112, 112))

    state_dict = torch.load(model_path)

    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name[:7].find('module.') >= 0:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value

    model.load_state_dict(new_state_dict)

    model.eval()
    return model


def forward_func(img_path, model):
    img_cv = cv2.imread(img_path)
    img_tensor = torch.from_numpy(img_cv.transpose((2, 0, 1))).float().unsqueeze(0)
    out = model(img_tensor)
    return out


if __name__ == '__main__':
    model = init_model()

    img_path = 'resources/cropped_face/test.bmp'

    output = forward_func(img_path, model)

    print(output, output.shape)
