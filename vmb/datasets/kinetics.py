import pickle
from os.path import join, isfile

from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.folder import find_classes, make_dataset

from vmb.datasets.presets import VideoClassificationPresetTrain, VideoClassificationPresetEval
import vmb.utils.logging as logging

logger = logging.get_logger(__name__)


class Kinetics(Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
        ], "Split '{}' not supported for Kinetics".format(mode)

        self.cfg = cfg
        self.split_folder = join(cfg.DATA.PATH_TO_DATA_DIR, mode)
        extensions = ("mp4", "avi")
        classes, class_to_idx = find_classes(self.split_folder)
        self.samples = make_dataset(directory=self.split_folder,
                                    class_to_idx=class_to_idx,
                                    extensions=extensions)
        video_list = [x[0] for x in self.samples]
        logger.info(f"Constructing {mode} dataset with {len(self.samples)} videos")

        saved_metadata = join(cfg.DATA.PATH_TO_DATA_DIR, F"{mode}_metadata.pkl")
        metadata = None
        if isfile(saved_metadata) and not cfg.DATA.OVERWRITE_METADATA:
            logger.info(f"Loading metadata (timestamps, fps) from {saved_metadata}")
            with open(saved_metadata, "rb") as f:
                metadata = pickle.load(f)

        self.video_clips = VideoClips(video_list,
                                      clip_length_in_frames=cfg.DATA.NUM_FRAMES,
                                      frames_between_clips=cfg.DATA.FRAMES_BETWEEN_CLIPS,
                                      frame_rate=cfg.DATA.FRAME_RATE,
                                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                                      _precomputed_metadata=metadata)
        if metadata is None:
            logger.info(f"Saving metadata (timestamps, fps, full video path) to {saved_metadata}")
            with open(saved_metadata, "wb") as f:
                pickle.dump(self.video_clips.metadata, f)
        if mode == "train":
            self.transform = VideoClassificationPresetTrain(min_max_scale_size=cfg.DATA.TRAIN_MIN_MAX_SCALE,
                                                            crop_size=cfg.DATA.TRAIN_CROP_SIZE,
                                                            mean=cfg.DATA.MEAN,
                                                            std=cfg.DATA.STD,
                                                            )
        elif mode == 'val':
            self.transform = VideoClassificationPresetEval(cfg.DATA.TEST_MIN_SHORT_SIDE_SCALE,
                                                           cfg.DATA.TEST_CROP_SIZE,
                                                           mean=cfg.DATA.MEAN,
                                                           std=cfg.DATA.STD,
                                                           )

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, _, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)

        return video, label
