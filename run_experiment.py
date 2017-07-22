

from fretFA import Experiment, SampleImage
from utils import *

path_bundle = request_paths()

experiment = Experiment(experiment_name = 'Test01',
                        paths_list = path_bundle,
                        b=10,
                        min_intensity_percentile = 0.05,
                        merger_threshold = 15,
                        min_segment_size = 5,
                        FA_segmentation_threshold = 750.0)

for sample in experiment.samples_dict:
    print('='*80)
    print(sample)
    for img in experiment.samples_dict[sample]:
        print('-'*80)
        print(img)
        image_object = SampleImage(experiment = experiment,
                                   sample_path = sample,
                                   filename = img)
        experiment.fret_df = experiment.fret_df.append(image_object.img_fret_df, ignore_index=True)

experiment.fret_df.to_csv(experiment.experiment_name+str('.csv'))
