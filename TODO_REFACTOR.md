* Each layer has both a forward and reverse
* OrientedPowermapModule: 
    * conv1x1 > conv3x3 gabor power > bn > relu > max_pool2d > conv1x1 > relu
    * as bottleneck module: shortcut: max_pool2d > conv1x1 to final channels > bn
    * reverse is: convnxn > bn > upsample, with in_channels/out_channels provided
    * can add l1 or bce loss function based on match between output and incoming input
        