from DINO.collect_dino_features import *
import cv2

def preprocess_frame(img, cfg):
    #if cfg['desired_height'] > 480 or cfg['desired_width']>640:
    #print("cfg['desired_width'],cfg['desired_height']", cfg['desired_width'],cfg['desired_height'] )
    return preprocess_image(img, half = cfg['use_16bit'],reshape_to = (cfg['desired_width'], cfg['desired_height'] ))

    #return preprocess_image(img, half = cfg['use_16bit'],reshape_to = (640, 480))
   
def get_dino_pixel_wise_features_model(cfg, device):

    ## See DINO.collect_dino_features

    # 添加参数默认值

    model = torch.hub.load(
        'facebookresearch/dino:main', 
        'dino_vitb8',
        pretrained=True,
        force_reload=False
    ).to(device).eval()
    # 初始化特征提取器
    feature_extractor = VITFeatureExtractor(
        backbone=model,  # 直接传入DINO模型
        upsample=True,
        stride=cfg.get('dino_strides', 8),
        desired_height=cfg.get('desired_height', 256),
        desired_width=cfg.get('desired_width', 455),
        multi_scale=[0.5, 1.0, 2.0]
    ).to(device)

 

    # 精度设置
    if cfg.get('use_16bit', False):
        feature_extractor = feature_extractor.half()

    # 参数冻结
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # 模型追踪（可选）
    if cfg.get('use_traced_model', False):
        example_input = torch.randn(1, 3, cfg['desired_height'], cfg['desired_width']).to(device)
        if cfg['use_16bit']:
            example_input = example_input.half()
        feature_extractor = torch.jit.trace(feature_extractor, example_input)

    return feature_extractor
