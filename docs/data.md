# Data

No videos, images or features are included. The standard benchmarks have their own terms, and you must request access from each:

- FaceForensics++: https://github.com/ondyari/FaceForensics (access form)
- Celeb-DF (v2): https://github.com/yuezunli/celeb-deepfakeforensics (access form)
- DFDC: https://ai.meta.com/datasets/dfdc/ (terms of use)

Arrange the videos as `<videos>/{train,val,test}/{real,fake}/*.mp4`. Split by **source identity**, not by clip: the same person must never appear in both train and test, or the test score will be inflated.

`python -m dfmamba.extract --videos <videos> --out <features>` samples 30 frames uniformly from each video, crops the largest face with a 25% margin, using the first available detector: OpenCV's YuNet if `DFM_YUNET` points to its `.onnx` file, then MediaPipe, then OpenCV 4's Haar cascade (removed in OpenCV 5), resizes it to 224 x 224, normalises it with ImageNet statistics in RGB order, runs the frozen EfficientNet-B0 and saves `{split}.pt`:

- `feats`: float16 [clips, 30, 1280]
- `labels`: 0 = real, 1 = fake
- `names`: the source file of each clip
- `skipped`: videos where no face was found (they are left out, so report how many per class)
- `faces_found`: how many of the 30 frames had a face; missing frames repeat the last face

Check that `faces_found` has a similar distribution for real and fake clips (extract prints the means). If face detection fails more often on fakes, repeated frames become a shortcut the temporal model can learn instead of real manipulation traces.

## Image datasets

Image datasets laid out as `<root>/{Train,Validation,Test}/{Real,Fake}/*.jpg` (any capitalisation; images already cropped to faces) are handled by `python -m dfmamba.extract_images --images <root> --out <features>`. Each image becomes a clip of one frame.
