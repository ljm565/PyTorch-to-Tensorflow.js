import os
import onnx
import numpy as np
import onnxruntime
import tensorflow as tf
from argparse import ArgumentParser
from onnx_tf.backend import prepare
from scc4onnx import order_conversion

import torch

from utils.config import Config
from models.linear import LN
from models.pplcnet import PPLCNet
from models.resnet import ResNet, ResNetTiny
from models.efficientformer2 import EfficientFormer2
from models.depthwiseseparable import DepthWiseSeparableNet



def torchOnnx_sanity_check(x, torch_model, onnx_model_path):
    torch_out = torch_model.forward(torch.FloatTensor(x)).detach().numpy()
    session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_out = session.run(['output'], {'input': x})
    print('torch: {}'.format(torch_out))
    print('onnx: {}'.format(onnx_out[0]))
    # assert np.sum(np.abs(torch_out - onnx_out)) < 1e-1
    return onnx_out


def onnxTf_sanity_check(x, onnx_out, tf_model_path):
    x = x.transpose(0, 2, 3, 1)
    tfmodel = tf.saved_model.load(tf_model_path)
    tf_out = tfmodel(input=tf.convert_to_tensor(x))['output']
    print('tf: {}'.format(tf_out))
    # assert np.sum(np.abs(onnx_out - tf_out)) < 1e-1


def main(args):
    model_name = args.name
    model_folder_path = './outputs/' + model_name + '/'
    config = Config(model_folder_path + model_name + '.json')
    size  = config.img_size
    check_point = torch.load(model_folder_path + model_name + '.pt', map_location=torch.device('cpu'))
    
    # torch to onnx
    onnx_path = model_folder_path + 'torchToOnnx.onnx'
    if 'resnet_small' in args.name:
        model = ResNet(config)
        model.load_state_dict(check_point['model'])
    elif 'resnet_tiny' in args.name:
        model = ResNetTiny(config)
        model.load_state_dict(check_point['model'])
    elif 'efficientformer' in args.name:
        model = EfficientFormer2(config)
        model.load_state_dict(check_point['model'])
    elif 'linear' in args.name:
        model = LN(config)
        model.load_state_dict(check_point['model'])
    elif 'pplcnet' in config.model:
        model = PPLCNet()
        model.load_state_dict(check_point['model'])
    elif 'depthwise-separable' in config.model:
        model = DepthWiseSeparableNet(config)
        model.load_state_dict(check_point['model'])

    model.eval()
    x = torch.randn(1, 3, size, size)
    torch.onnx.export(model, x, onnx_path, verbose=True, input_names=['input'], output_names=['output'])


    # check sanity check between torch and onnx
    x = np.random.randn(1, 3, size, size)
    x = x.astype(np.float32)
    onnx_out = torchOnnx_sanity_check(x, model, onnx_path)
    


    # onnx to tf
    ## channel conversion
    onnx_model = onnx.load(onnx_path)
    input_name = onnx_model.graph.input[0].name
    onnx_model = order_conversion(
        onnx_graph=onnx_model,
        input_op_names_and_order_dims={f"{input_name}": [0,2,3,1]},
        non_verbose=True
    )

    tf_rep = prepare(onnx_model)
    tf_path = model_folder_path + 'tfModel'
    tf_rep.export_graph(tf_path)


    # sanity check between onnx and tensorflow
    onnxTf_sanity_check(x, onnx_out, tf_path)


    # tf to tfjs
    cmd = 'tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve'
    cmd = cmd + ' ' + tf_path
    cmd = cmd + ' ' + model_folder_path + 'tfjsModel_' + args.name
    os.system(cmd)    




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True)
    args = parser.parse_args()


    main(args)