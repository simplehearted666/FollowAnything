# FollowAnything
*解决了torch.cuda.amp.autocast 弃用、调用 torch.meshgrid 时未传入 indexing 参数以及使用 cv::imwrite 保存图像时，输入图像的深度不被所选编码器支持的问题*
## Windows系统环境配置：
1. python下载：[python3.9.12](https://www.python.org/downloads/release/python-3912/)

2. 查看可与电脑GPU兼容的CUDA并安装

    - cmd查询GPU计算能力
      ```
      nvidia-smi --query-gpu=compute_cap --format=csv
      ```

    - 官网下载CUDA ToolKit：[CUDA ToolKit](https://developer.nvidia.com/cuda-toolkit-archive) 

    -  官网下载CUDA Drivers: [CUDA Drivers](https://www.nvidia.com/en-us/drivers/)

    - 环境变量配置：[CUDA环境变量](https://wenku.csdn.net/answer/5dfjqtvp2x)
   
    | CUDA 版本 | 最低驱动版本 | 典型支持的 GPU 架构（示例） | 计算能力范围 |
    |----------|-------------|----------------------------|--------------|
    | 12.x     | 525.60+     | Ada Lovelace, Hopper        | 5.0 – 9.0+   |
    | 11.x     | 450.80+     | Ampere, Turing              | 3.5 – 8.6    |
    | 10.x     | 410.48+     | Volta, Pascal               | 3.0 – 7.5    |



3. 安装与CUDA版本兼容的torch和torchvision

    - 官网下载PyTorch：[PyTorch](https://pytorch.org/get-started/locally/)

  
    | PyTorch 版本 | 支持 CUDA 版本 | 
    |--------------|---------------|
    |     2.3+     |	 12.1~12.4	 |
    |    2.0~2.2   |   11.7~12.1   |	
    |   1.13~1.15  |   11.6~11.8   |

4. 下载模型文件(安装路径为`FollowAnything\Segment-and-Track-Anything\ckpt`)

    - SAM模型：[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
  
    - AOT模型：[ R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.usercontent.google.com/download?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ&export=download)

5. 开发工具安装

    - 下载Visual Studio 2019   [Visual Studio 2019](https://link.zhihu.com/?target=https%3A//aka.ms/vs/16/release/vs_community.exe)   (**确保已安装 "使用 C++ 的桌面开发" 和 "Windows 10 SDK"**)
  
      
6. 依赖库安装

    - 安装核心依赖：
      ```
      pip install numpy==1.23.5 opencv-python matplotlib scipy tqdm scikit-image mavsdk
      ```

        * 若速度较慢，可选用镜像源下载：
        ```
        pip install numpy==1.23.5 opencv-python matplotlib scipy tqdm scikit-image mavsdk -i https://mirrors.aliyun.com/pypi/simple
        ```
   - 处理兼容性依赖：
     ```
     pip install spatial-correlation-sampler
     ```

     * 常见问题：安装时提示 **OSError: CUDA_HOME not set**，[解决方案](https://developer.baidu.com/article/details/3269640)


7. 运行示例命令
   ```
   python follow_anything.py --desired_height 240 --desired_width 320 --path_to_video example_videos/brick_following.avi --save_images_to outputs/ --detect dino --use_sam --tracker aot --queries_dir queries/brick_following/ --desired_feature 7 --plot_visualizations
   ```

     
      

  




