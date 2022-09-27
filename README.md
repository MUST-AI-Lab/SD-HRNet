# SD-HRNet

## Quick start

### Test
````bash
python tools/test.py --cfg <CONFIG-FILE> --model-file <MODEL WEIGHT> --mask-file <SUPERNET BETA WEIGHT>
# example:
python tools/test.py --cfg experiments/300w/face_alignment_300w_shrinking_hrnet_mb.yaml --model-file model.pth --mask-file mask.pth --seed 111
````
