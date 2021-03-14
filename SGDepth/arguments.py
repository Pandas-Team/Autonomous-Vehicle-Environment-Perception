from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class ArgumentsBase(object):
    def __init__(self):
        self.ap = ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter
        )

    def running_args(self):
        self.ap.add_argument('--weights-detector', type=str, default='weights/yolov5m.pt', help='weights for the main yolo detector')
        self.ap.add_argument('--weights-sign'    , type=str, default='weights/best_sign2.pt', help='sign detector weights')
        self.ap.add_argument('--disp-detector', type=str, default='weights/model_full.pth', help='disparity model weights')
        self.ap.add_argument('--lane-detector-type', type=str, default='culane', help='Choose between culane or curvelane')
        self.ap.add_argument('--culane-model', type=str, default='weights/culane_model.pkl', help='Culane model')
        self.ap.add_argument('--curvelane-model', type=str, default='weights/curvelane_model.pkl', help='Curvelane model')
        self.ap.add_argument('--video', type=str, default='test.mov', help = 'The input video')
        self.ap.add_argument('--rotate', action = 'store_true', default='Rah.mov', help = 'The input video')
        self.ap.add_argument('--save', action= 'store_true', help = 'Saving the output video')
        self.ap.add_argument('--noshow', action= 'store_true', help =  'Do not Show the output frames')
        self.ap.add_argument('--frame-drop', type = int, default = 1 , help =  'Frame Drop for processing')
        self.ap.add_argument('--outputfps', type = int, default = 30 , help =  'Output Video FPS')
        self.ap.add_argument('--fps', action = 'store_true' , help =  'Show fps')
        self.ap.add_argument('--output-name', type = str ,default = 'output.mov' , help =  'Outputput video address')
        self.ap.add_argument('--depth-mode', type = str ,default = 'kitti' , help =  'Choosing depth mode (kitti or cityscape)')
        self.ap.add_argument('--mode', type = int, default = 1, help = 'Choose theprocessing model (1,2,3)')
        self.ap.add_argument('--save-frames', action = 'store_true' , help =  'Saves individual Frames')


    def SGDepth_harness_init_system(self):
        self.ap.add_argument(
            '--sys-cpu', default=False, action='store_true',
            help='Disable Hardware acceleration'
        )

        self.ap.add_argument(
            '--sys-num-workers', type=int, default=3,
            help='Number of worker processes to spawn per DataLoader'
        )

        self.ap.add_argument(
            '--sys-best-effort-determinism', default=False, action='store_true',
            help='Try and make some parts of the training/validation deterministic'
        )


    def SGDepth_harness_init_model(self):

        self.ap.add_argument(
            '--model-num-layers', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the depth and segmentation encoder'
        )

        self.ap.add_argument(
            '--model-num-layers-pose', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the pose encoder'
        )

        self.ap.add_argument(
            '--model-split-pos', type=int, default=1, choices=(0, 1, 2, 3, 4),
            help='Position in the decoder to split from common to separate depth/segmentation decoders'
        )

        self.ap.add_argument(
            '--model-depth-min', type=float, default=0.1,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-max', type=float, default=100.0,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-resolutions', type=int, default=4, choices=(1, 2, 3, 4),
            help='Number of depth resolutions to generate in the network'
        )

        self.ap.add_argument(
            '--experiment-class', type=str, default='sgdepth_eccv_test',
            help='A nickname for the experiment folder'
        )

        self.ap.add_argument(
            '--model-name', type=str, default='sgdepth_base',
            help='A nickname for this model'
        )

        self.ap.add_argument(
            '--model-load', type=str, default=None,
            help='Load a model state from a state directory containing *.pth files'
        )

        self.ap.add_argument(
            '--model-disable-lr-loading', default=False, action='store_true',
            help='Do not load the learning rate scheduler if you load a checkpoint'
        )


    def _parse(self):
        return self.ap.parse_args()


class InferenceEvaluationArguments(ArgumentsBase):
    def __init__(self):
        super().__init__()
        self.running_args()
        self.SGDepth_harness_init_system()
        self.SGDepth_harness_init_model()

    def parse(self):
        opt = self._parse()

        return opt
