from re import I
import onnx
import tensorflow as tf
import onnxruntime
from onnx_tf.backend import prepare
from onnxruntime.quantization import quantize
from onnxruntime.quantization.quant_utils import QuantizationMode
import random

import os
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import torch
from models.face_model import FaceNet,FaceNet_Mid, FaceNet_v2
from models.enhance_model import EnhanceNet,EnhanceNetX1,EnhanceNetX2, EnhanceNetX2_v2,EnhanceNetX3,EnhanceNet_x3
from models.anime_model import AnimeNet,AnimeNet2,AnimeNet4
import utils
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.backends._nnapi import prepare as nnapi_prepare
import torch.quantization.quantize_fx as quantize_fx
import copy

from utils import utils_image as util
import cv2




torch_model_scripted_vulkan_path = os.path.join('model_zoo', 'BSRGAN_scripted_vulkan.pth')
torch_model_quant_scripted_lite_path = os.path.join('model_zoo', 'BSRGAN_scripted.pth')
torch_model_path = os.path.join('model_zoo', 'BSRGAN.pth') 
torch_model_quant_path = os.path.join('model_zoo', 'BSRGAN_quant.pth') 
torch_model_static_quant_graph_path = os.path.join('model_zoo', 'BSRGAN_static_quant_graph.pth') 
torch_model_quant_dynamic_path = os.path.join('model_zoo', 'BSRGAN_quant_graph_dynamic.pth') 
torch_model_quant_graph_dynamic_path = os.path.join('model_zoo', 'BSRGAN_quant_graph_dynamic.pth') 
torch_model_quant_dynamic_scripted_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted.pth') 
torch_model_scripted_nnapi_path = os.path.join('model_zoo', 'BSRGAN_scripted_nnapi.pth') 
torch_model_quant_scripted_vulakn_path = os.path.join('model_zoo', 'BSRGAN_quant_scripted_vulkan.pth') 
onnx_path = os.path.join('model_zoo', 'BSRGAN_ONNX.onnx')        
onnx_quant_path = os.path.join('model_zoo', 'BSRGAN_ONNX_quant.onnx')        
onnx_quant_static_path = os.path.join('model_zoo', 'BSRGAN_ONNX_quant_static.onnx')        
tf_rep_path = os.path.join('model_zoo', 'BSRGAN_tf_Rep')         
tf_lite_path = os.path.join('model_zoo', 'BSRGAN_tf_Lite')         
device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
L_path = os.path.join("testsets", 'test_data')
D_path = os.path.join("testsets", 'test')

print(device)

# def test_with_image(model,OUT_NAME,dtype = torch.float32):
#     model.to(device)
#     with torch.no_grad():
#         for img in util.get_image_paths(L_path):
#             if('bakbak' in img):
#                 torch.cuda.empty_cache()
#                 img_L = util.imread_uint(img, n_channels=3)
#                 if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
#                     print("img{}".format(np.shape(img_L)))
#                     img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).div(255.).unsqueeze(0)
#                     img_L = img_L.type(dtype)
#                     img_L = img_L.to(device)
#                     start = time.time()
#                     img_E = model(img_L)
#                     print("inference time: {}".format(time.time() - start))
#                     img_E = util.tensor2uint(img_E)
#                     util.imsave(img_E, os.path.join('test/', OUT_NAME+'.png'))

#     model.cpu()


def test_with_image(model,OUT_NAME,dtype = torch.float32):
    model.to(device)
    with torch.no_grad():
        for img in util.get_image_paths(L_path):
            if('1646570299313' in img):
                torch.cuda.empty_cache()
                img_L = util.imread_uint(img, n_channels=3)
                w= np.shape(img_L)[1]
                h = np.shape(img_L)[0]
                img_L = cv2.line(img_L,pt1 = (700, 500), pt2 = (800, 300),
                            color = (0, 0, 0),thickness = random.randint(5,20))
                img_L =  cv2.circle(img_L,center = (350, 278),radius = 20,color = (0, 0, 0),thickness = -1)
                img_L =  cv2.circle(img_L,center = (330, 278),radius = 20,color = (0, 0, 0),thickness = -1)
                img_L =  cv2.circle(img_L,center = (310, 278),radius = 20,color = (0, 0, 0),thickness = -1)
                img_L = cv2.line(img_L,pt1 = (500, 600), pt2 = (600, 800),color = (0, 0, 0),thickness = 30)
                util.imsave(img_L, os.path.join('testsets/exported', 'input'+'.png'))
            
                #if(np.shape(img_L)[0] < 1055 and np.shape(img_L)[1] < 1055):
                print("img{}".format(np.shape(img_L)))
                img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).div(255.).unsqueeze(0)
                img_L = img_L.type(dtype)
                img_L = img_L.to(device)
                start = time.time()
                img_E = model(img_L)
                print("inference time: {}".format(time.time() - start))
                img_E = util.tensor2uint(img_E)
                util.imsave(img_E, os.path.join('testsets/exported', OUT_NAME+'.png'))

    model.cpu()




#LOAD TORCH MODEL
main_path = "checkpoints/face_net_checkpoints/black_mask_checkpoints/v3"
toch_model_path = "checkpoints/face_net_checkpoints/black_mask_checkpoints/v3/checkpoint_base_epoch_53.pth"#"/home/anis/Desktop/AI/MultiSPE/checkpoints/face_net_checkpoints/checkpoint_base_epoch_19.pth"#os.path.join('checkpoints/face_net_checkpoints', 'checkpoint_base_epoch_19.pth')
torch_model = FaceNet_v2()
        
#torch_model = architecture.IMDN(upscale=4)
checkpoint = torch.load(toch_model_path)
torch_model.load_state_dict(checkpoint["model_base_state_dict"])
torch_model.eval()
torch_model.to(device)
test_with_image(torch_model,'output')
# #torch_model = torch_model.to(device)

# #Post Training Static Quantization
# # # modules_to_fuse = [['upconv1', 'lrelu'],
# # #                    ['upconv2', 'lrelu'],
# # #                     ['HRconv', 'lrelu'],
# # #                  ]
# # # model_f32_fused = torch.quantization.fuse_modules(torch_model, modules_to_fuse, inplace=True)
# torch_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
# torch.backends.quantized.engine = 'qnnpack'
# model_fp32_prepared =torch.quantization.prepare(torch_model)

# model_fp32_prepared.to(device)

# with torch.no_grad():
#     for img in util.get_image_paths(L_path):
#         torch.cuda.empty_cache()

#         img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
#         if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
#             img_L = util.uint2tensor4(img_L)#return pytorch tensor with 1*C*H*W
#             img_L = img_L.to(device)
#             model_fp32_prepared(img_L)
#             break
            
            
        
# model_fp32_prepared.eval()    
# model_fp32_prepared.cpu()        
# torch_model =torch.quantization.convert(model_fp32_prepared)    



# # #FX Graph Mode Quantization
# # # torch_model.to(device)
# # # model_to_quantize = copy.deepcopy(torch_model)
# # # qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
# # # model_to_quantize.eval()
# # # # prepare
# # # model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
# # # # calibrate (not shown)
# # # model_prepared.to(device)
# # # L_path = os.path.join("testsets", 'RealSRSet')

# # # with torch.no_grad():
# # #     for img in util.get_image_paths(L_path):
# # #         torch.cuda.empty_cache()

# # #         img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
# # #         if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
# # #             img_L = util.uint2tensor4(img_L)#return pytorch tensor with 1*C*H*W
            
# # #             img_L = img_L.to(device)
# # #             model_prepared(img_L)
            
# # # # quantize
# # # model_quantized = quantize_fx.convert_fx(model_prepared)
# # # input_tensor = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
# # # print(model_quantized(input_tensor))

# # # torch.save(model_quantized.state_dict(), torch_model_quant_graph_path)


# torch_model = torch.quantization.quantize_dynamic(
#     torch_model, {torch.nn.Conv2d}, dtype=torch.qint8)
# test_with_image(torch_model,'quantizated_dynamic_test')
# torch.save(quantized_model.state_dict(),os.path.join('model_zoo','BSRGAN_dynamic_quantiated_model.pth'))

# # #dynamic/weight_only Quantization
# # # model_to_quantize = copy.deepcopy(torch_model)
# # # model_to_quantize.eval()
# # # torch_model.to(device)
# # # qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
# # # # prepare
# # # model_prepared = quantize_fx.prepare_fx(torch_model, qconfig_dict)
# # # # no calibration needed when we only have dynamici/weight_only quantization
# # # model_prepared.to(device)
# # # input_tensor = None
# # # with torch.no_grad():
# # #     for img in util.get_image_paths(L_path):
# # #         torch.cuda.empty_cache()

# # #         img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
# # #         if(np.shape(img_L)[0] < 512 and np.shape(img_L)[1] < 512):
# # #             img_L = util.uint2tensor4(img_L)#return pytorch tensor with 1*C*H*W
# # #             input_tensor = img_L
# # #             img_L = img_L.to(device)
# # #             model_prepared(img_L)
# # #             break
            
        
# # # model_prepared.eval()    
# # # model_prepared.cpu()        
# # # # quantize
# # # model_quantized = quantize_fx.convert_fx(model_prepared)
# # # scripted_model = torch.jit.script(model_quantized)
# # # torch.jit.save(scripted_model,torch_model_scripted_path)
# # # img_E = model_quantized(input_tensor)
# # # img_E = util.tensor2uint(img_E)
# # # util.imsave(img_E, os.path.join('test/', 'test_dynamic.png'))



# #test model
# #torch_model.load_state_dict(torch.load(os.path.join('model_zoo','BSRGAN_static_quantiated_model.pth')))
# model_int8_quantized.to(device)
# test_with_image(torch_model,"static_quantized_torch_model",dtype=torch.float32)



# # # torch_model.to(device)

# # #Trace
# # # torch.cuda.empty_cache()
# # torch_model.to(device)
# # traced_model = torch.jit.trace(torch_model,torch.rand(1, 3, 40, 40).to(device))
# # test_with_image(traced_model,"traced_model")
# # torch.jit.save(traced_model,os.path.join('model_zoo','BSRGAN_traced_model.pth'))

# #save for lite interpreted
# # # #TORCHSCRIPT
#scripted_model = torch.jit.load(os.path.join('model_zoo','BSRGAN_scripted_static_quantized_model.pth'))
# scripted_model._save_for_lite_interpreter(os.path.join('model_zoo','BSRGAN_lite_static_quantized_model.pth'))
# scripted_model_optimized = optimize_for_mobile(scripted_model,backend="cpu")
# scripted_model_optimized._save_for_lite_interpreter(os.path.join('model_zoo','BSRGAN_cpu_lite_static_quantized_model.pth'))
# scripted_model_optimized = optimize_for_mobile(scripted_model,backend="Vulkan")
# scripted_model_optimized._save_for_lite_interpreter(os.path.join('model_zoo','BSRGAN_vulkan_lite_static_quantized_model.pth'))

# scripted_torch_model = torch.jit.script(torch_model)
# scripted_model_optimized = optimize_for_mobile(scripted_torch_model,backend="cpu")
# scripted_model_optimized._save_for_lite_interpreter(os.path.join('checkpoints/face_net_checkpoints/black_mask_checkpoints/v3/script','lite_cpu_facenet.pth'))
# scripted_model_optimized = optimize_for_mobile(scripted_torch_model,backend="Vulkan")
# scripted_model_optimized._save_for_lite_interpreter(os.path.join('checkpoints/face_net_checkpoints/black_mask_checkpoints/v3/script','lite_vulkan_facenet.pth'))

# # #to NNAPI 
# # scripted_model = torch.jit.script(model_int8_quantized)
# # input_tensor = torch.from_numpy(np.random.randn(1, 3, 50, 50).astype(np.float32))
# # input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
# # input_tensor.nnapi_nhwc = True
# # nnapi_model = nnapi_prepare.convert_model_to_nnapi(scripted_model,input_tensor)
# # nnapi_model._save_for_lite_interpreter(torch_model_scripted_nnapi_path)


# #TORCH TO ONNX

def exportToOnnx(model,input,output,path):
    print("export the torch model to "+path)
    # Export the model
    torch.onnx.export(model,               # model being run
                        input,                         # model input (or a tuple for multiple inputs)
                        path,   # where to save the model (can be a file or file-like object)
                        example_outputs=output,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={
                                    'input' : {2 : 'inputc_h', 3: 'inputc_w'},
                                    'output' : {2 : 'output_h', 3: 'output_w'},
                                    }
                                    );
    print("finished exporting to onnx")

print("mode loaded to {}".format(device))
img_L = torch.from_numpy(np.random.randn(1,3, 250, 250).astype(np.float32))
img_E = torch.from_numpy(np.random.randn(1,3, 250*1, 250*1).astype(np.float32))
exportToOnnx(model = torch_model,input = img_L,output = img_E,path = os.path.join(main_path+'/script', 'face_net_onnx.pth'))





# # # # Load the ONNX model
onnx_path = os.path.join(main_path+'/script', 'face_net_onnx.pth')
# print("load onnx")
# model_onnx = onnx.load(onnx_path)

# # # #Check that the IR is well formed
# onnx.checker.check_model(model_onnx)
# # # #Print a Human readable representation of the graph
# print(onnx.helper.printable_graph(model_onnx.graph))
# # quantized_model = quantize(model_onnx,
# #                             quantization_mode=QuantizationMode.IntegerOps,
# #                             symmetric_weight=True,
# #                             force_fusions=True)

# # onnx.save(quantized_model, onnx_quant_path)

# print("test onnx runtime")
# ort_session = onnxruntime.InferenceSession(onnx_path)

# outputs = ort_session.run(
#     None,
#     {'input': np.random.randn(1, 3, 80, 80).astype(np.float32)}
# )
# print(np.shape(outputs))


tf_rep_path = os.path.join(main_path+'/script', 'face_net_tf_rep')
# #ONNX to TF
model_onnx = onnx.load(onnx_path)
print("convert onnx to tensorflow representation")
tf_rep = prepare(model_onnx)    
tf_rep.export_graph(tf_rep_path)



# #TF Model Inference
print("test tf represenation inference")
model_tf = tf.saved_model.load(tf_rep_path)
model_tf.trainable = False

input_tensor = tf.random.uniform([1, 3, 40, 40])
out = model_tf(**{'input': input_tensor})
print(out["output"].shape)



tf_lite_path = os.path.join(main_path+'/script', 'facenet.tflite')
#TF to TFLite
print("Convert the tf_rep model to tflite")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep_path)


tflite_model = converter.convert()

# Save the model
print("save tf lite model to ")
with open(tf_lite_path, 'wb') as f:
    f.write(tflite_model)


# # #optimization
# def rep_data_gen():
#     # for img in util.get_image_paths(L_path):
#     #     print(img)
#     #     img_L = cv2.imread(img.data, cv2.IMREAD_UNCHANGED)  # BGR or G
#     #     img_L = util.imread_uint(img_L, n_channels=3)#return numpy array RGB H*W*
#     #     img_L = np.ascontiguousarray(img_L)
#     #     img_L = np.transpose(img_L, (2, 0, 1)).astype(np.float32)/255.
#     #     yield [img_L]

#     import struct
#     imgs = tf.data.Dataset.from_tensor_slices(util.get_image_paths(D_path)).batch(1)
#     for img in imgs:
#         img_L = cv2.imread(img.numpy()[0].decode(), cv2.IMREAD_UNCHANGED)  # BGR or G
#         #img_L = util.imread_uint(img_L, n_channels=3)#return numpy array RGB H*W*
#         img_L = np.ascontiguousarray(img_L)
#         img_L = np.transpose(img_L, (2, 0, 1))/255. #RGB
#         img_L = np.expand_dims(img_L,0).astype(np.float32)
#         print("image {}".format(np.shape(img_L)))
#         yield [img_L]


# converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep_path)
# #Convert using dynamic range quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# #converter.target_spec.supported_types = [tf.float16]
# # converter.representative_dataset = rep_data_gen
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # converter.inference_input_type = tf.uint8
# # converter.inference_output_type = tf.uint8
# tflite_quant_model = converter.convert()
# # Save the quantizated model
# with open(tf_lite_path+"dynamic_quant.tflite", 'wb') as f:
#     f.write(tflite_quant_model)

#TFLite Model Inference
#Load the TFLite model and allocate tensors
# print("test tf lite inference")
# interpreter = tf.lite.Interpreter(model_path=tf_lite_path)

# input_details = interpreter.get_input_details()

# # Test the model on random input data
# img_L = None
# for img in util.get_image_paths(L_path):
#             if('bakbak' in img):
#                 torch.cuda.empty_cache()
#                 img_L = util.imread_uint(img, n_channels=3)#return numpy array RGB H*W*C
#                 img_L = np.ascontiguousarray(img_L)/255.
#                 img_L = np.transpose(img_L,(2, 0, 1))
#                 img_L = np.expand_dims(img_L,0)
#                 img_L = img_L.astype(np.float32)
           
                


# #resize inputs to match model inputs
# interpreter.resize_tensor_input(
#     input_details[0]['index'], img_L.shape)
# interpreter.allocate_tensors()
# # Get input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
# print("input shape {}".format(input_shape))



# interpreter.allocate_tensors()
# interpreter.set_tensor(input_details[0]['index'], img_L)

# start = time.time()
# interpreter.invoke()
# print("inference time {}".format(time.time()- start))
# # get_tensor() returns a copy of the tensor data
# # use tensor() in order to get a pointer to the tensor
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("output shape {}".format(np.shape(output_data)))
# output_data = np.squeeze(output_data)
# output_data = np.clip(output_data,0,1)
# output_data = np.transpose(output_data, (1, 2, 0))
# output_data = np.uint8((output_data*255.0).round())
# util.imsave(output_data, os.path.join('test/', 'ESRGAN test'+'.png'))




