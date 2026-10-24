"""Three actual synchronized L/F/R frames; labels never enter actor inputs."""
from PIL import Image
from .data import RawSceneData


class TemporalSceneData(RawSceneData):
    def __init__(self, root, *, cohort):
        super().__init__(root, cohort=cohort)
        if self.manifest['schema'] != 'native_qwen_sourceflow_temporal_scene_v1':
            raise ValueError('Explicit three-frame data identity required')
        if self.manifest['frame_offsets'] != [-2, -1, 0]:
            raise ValueError('Only two historical frames plus current are registered')

    def observation(self, token):
        row = self.records[token]
        frames = []
        for frame in row['observation_frames']:
            images = {}
            for camera, path in frame['images'].items():
                if path is None:
                    images[camera] = None
                else:
                    with Image.open(path) as image:
                        images[camera] = image.convert('RGB')
            frames.append(dict(images=images, timestamp_us=frame['timestamp_us'],
                relative_seconds=frame['relative_seconds'], offset=frame['offset']))
        if len(frames) != 3 or frames[-1]['relative_seconds'] != 0.:
            raise ValueError('Current plus exactly two historical frames required')
        return dict(frames=frames, images=frames[-1]['images'],
            proprio=list(row['proprio']), navigation=row['navigation'])
