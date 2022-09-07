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

According to the paper, I think these defined values are correct.  

The second different part is:

reshape operation is slow and sometimes not operated in order, So I changed it to rearrange, and matrix multiplication is performed using einsum.

The third different part is:

I implemented a StageModule class like Swin Transformer, thereby I can insert this module into any model architecture.

The building block of CoTNet was implemented (paper: Contextual Transformer Networks for Visual Recognition)

I made several changes to this model:

1. I used AdaptiveAveragePool2d to replace LocalConv to reduce computation cost, this part is similar to PVTv2 

2. The original CoTNet does not perform pixel-index attention: K2 = Q * V, this is only channel-index attention. So I added a MLP-mixer to do pixel-index attention in the code.

3. The reason for using SK attention for K1 and K2 fusion is not clear (paper does not explain that), so I deleted the SK attention and added a shortcut connection as a replacement.

The modification of outlook attention: outlook attention improves the fine-grained feature generation

1) the output shape of Unfold operation should be:

new_H = math.floor((H + 2 * self.padding - self.kernel_size + self.stride)/self.stride) 
        
new_W = math.floor((W + 2 * self.padding - self.kernel_size + self.stride)/self.stride)

2) using Conv2d(..., kernel_size=1, stride=1, padding=0, ...) to generate v and attn, which can save permute operation time.

3) replace LayerNorm with BatchNorm2d, because the weights of attn, v and MLP are generated by Conv2d. Therefore, it is reasonable to use BatchNorm2d not LayerNorm


In the end, if you have any question, please send email to me:

xingshuli600@gmail.com

