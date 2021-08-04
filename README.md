# Vision-In-Transformer-Model
Apply Transformer Models to Computer Vision Tasks

The implementation about relative position embedding can refer to:

https://theaisummer.com/positional-embeddings/: BoT position embedding method (refer to BoT_Position_Embedding.png and BoT_Position_Embedding(2).png)

Swin Transformer position embedding refer to Swin_Transformer_Position_Embedding.png

The implementation detail about Swin Transformer can refer to:

https://zhuanlan.zhihu.com/p/361366090

One Swin Transformer Block = Swin_Transformer_Stage

Swin_Transformer_Stage = Swin_Transformer + Patch_Merge


Pyramid Vision Transformer-v2 code finished (2021-08-04) refer to PVTv2_Block

I changed some parts of PVTv2 from the official released version:(https://github.com/whai362/PVT/blob/v2/classification/pvt_v2.py)

The first different part is: 

kernel_size = patch_size

stride = math.ceil(kernel_size / 2)

padding = math.floor(stride / 2)

According the paper, I think these defined values are correct.  

The second different part is:

reshape operation is slow and sometimes not operated in order, So I changed it to rearrange, and matrix multiplication is performed using einsum.

The third different part is:

I implemented a StageModule class like Swin Transformer, thereby I can insert this module into any model architecture.


paper reading now:

CoAtNet: Marrying Convolution and Attention for All Data Sizes (The idea of parameter reconstruction in ViT model)

ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases (Discussion about how to combine Transformer block into CNN)

Twins: Revisiting the Design of Spatial Attention in Vision Transformers (Discussion about local and global effect on model accuracy)

If you have any question, please send email to me:

xingshuli600@gmail.com
