import pickle
import torch

backbone_file = 'models/mobilenet_v2-7ebf99e0.pth'
head_file = 'models/edited_r50_coco.pkl'

with open(head_file, "rb") as f:
    head_dict = pickle.load(f)

head = head_dict['model']
new = head_dict['model'].copy()

backbone = torch.load(backbone_file)

# print(backbone.keys())
num_bn_keys = 0
for k in head.keys():
    if k.startswith("backbone"):
        new.pop(k)
        num_bn_keys += 1
    if k.startswith("sem_seg_head.pixel_decoder"):
        if k.startswith("sem_seg_head.pixel_decoder.input_proj"):
            new.pop(k)
        elif k.startswith("sem_seg_head.pixel_decoder.adapter"):
            new.pop(k)
        else:
            suffix = k.split("sem_seg_head.pixel_decoder")[-1]
            new.pop(k)
            new["sem_seg_head" + suffix] = head[k]


for k in backbone.keys():
    # subparams = k.split('.')
    # if len(subparams) >= 5:
    #     f, l, c, x, y = subparams[:5]
    #     n = x
    #     if x == 0 and x == 0:
    #         n = 0
    #     if x == 0 and y == 1:
    #         n = 1
    #     if x == 1 and y == 0:
    #         n = 3
    #     if x == 1 and y == 1:
    #         n = 4
    #     if x == 2:
    #         n = 6
    #     if x == 3:
    #         n == 7

    #     key_name = f"backbone.{f}.{l}.{c}.{n}.{subparams[-1]}"
    # else:
    key_name = "backbone." + k
    new[key_name] = backbone[k]
    

head_dict['model'] = new

# print(len(backbone.keys()), num_bn_keys)

with open("models/edited_mobilenet.pkl", "wb") as f:
    pickle.dump(head_dict, f)

# predictor_root = "sem_seg_head.predictor"
# subparams = ["class_embed", "decoder_norm", "mask_embed", "transformer_cross_attention_layers",
#             "transformer_ffn_layers", "transformer_self_attention_layers"]

# model = data["model"]
# model_keys = list(model.keys())

# for p in subparams:
#     key_prefix = f"{predictor_root}.{p}"
#     keys = [m for m in model_keys if m.startswith(key_prefix)]
#     for key in keys:
#         suffix = key.split(predictor_root)[-1]
#         for d in ["plant_decoder", "leaf_decoder"]:
#             print(f"{predictor_root}.{d}{suffix}")
#             model[f"{predictor_root}.{d}{suffix}"] = model[key]
#         if p != "decoder_norm":
#             model.pop(key)

# data["model"] = model

# with open("models/edited_swin_lin21k.pkl", "wb") as f:
#     pickle.dump(data, f)