Using TNN as readout for liquid

Lab 1 Notes:
- Intensity2Latency used for MNIST 
- Image Size [5, 5] // Size of receptive field
- Input Size [8, 2, 5, 5] // Time, ON/OFF Channel, Image Size
- Output Size [8, 32] // Num timesteps by num neurons
- Weights [8, 2, 5, 5, 32]

rTNN Notes:
- Output from reservoir is [250, 500] // Time x Num neurons
- Sample code just sums over first dimension and trains using scalars
- Input Size into column would be [250, 500] compared to [8, 2, 5, 5]
- Interesting avenue of research -> how to adapt input to be sent into column
