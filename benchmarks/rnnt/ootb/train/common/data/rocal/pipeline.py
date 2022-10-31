# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import nvidia.dali
# import nvidia.dali.ops as ops
# import nvidia.dali.types as types
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import multiprocessing
import numpy as np
import torch
import math

class PipelineParams:
    def __init__(
            self,
            sample_rate=16000,
            max_duration=float("inf"),
            normalize_transcripts=True,
            trim_silence=False,
            speed_perturbation=None
        ):
        pass

class SpeedPerturbationParams:
    def __init__(
            self,
            min_rate=0.85,
            max_rate=1.15,
            p=1.0,
        ):
        pass

class RocalPipeline(object):
    def __init__(self, *,
                 pipeline_type,
                 device_id,
                 num_threads,
                 batch_size,
                 file_root: str,
                 sampler,
                 sample_rate,
                 resample_range: list,
                 window_size,
                 window_stride,
                 nfeatures,
                 nfft,
                 dither_coeff,
                 silence_threshold,
                 preemph_coeff,
                 max_duration,
                 preprocessing_device="gpu"):

        self._rocal_init_log(locals())

        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            n_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        self.preprocessing_device = preprocessing_device.lower()
        assert self.preprocessing_device == "cpu" or self.preprocessing_device == "gpu", \
            "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"

        self.resample_range = resample_range

        train_pipeline = pipeline_type == 'train'
        
        self.train = train_pipeline
        if self.train :
            self.file_root_rocal="/dataset/Audio_Dataset/LibriSpeech/train-clean-100-wav/"
        else:
            self.file_root_rocal="/dataset/Audio_Dataset/LibriSpeech/test-clean-wav/"
            
        self.sample_rate = sample_rate
        self.dither_coeff = dither_coeff
        self.nfeatures = nfeatures
        self.max_duration = max_duration
        self.do_remove_silence = True if silence_threshold is not None else False
        self.batch_size = batch_size
        self.file_root = file_root
        self.num_threads = num_threads
        self.nfft = nfft
        self.preemph_coeff = preemph_coeff
        self.device_id = device_id
        self.sampler = sampler
        self.window_size = window_size
        self.window_stride = window_stride
        
        shuffle = train_pipeline and not sampler.is_sampler_random()

    @classmethod
    def from_config(cls, pipeline_type, device_id, batch_size, file_root: str, sampler, config_data: dict,
                    config_features: dict, device_type: str = "gpu", do_resampling: bool = True,
                    num_cpu_threads=multiprocessing.cpu_count()):

        max_duration = config_data['max_duration']
        sample_rate = config_data['sample_rate']
        silence_threshold = -60 if config_data['trim_silence'] else None

        if do_resampling and config_data['speed_perturbation'] is not None:
            resample_range = [config_data['speed_perturbation']['min_rate'],
                              config_data['speed_perturbation']['max_rate']]
        else:
            resample_range = None

        window_size = config_features['window_size']
        window_stride = config_features['window_stride']
        nfeatures = config_features['n_filt']
        nfft = config_features['n_fft']
        dither_coeff = config_features['dither']
        preemph_coeff = .97

        return cls(pipeline_type=pipeline_type,
                   device_id=device_id,
                   preprocessing_device=device_type,
                   num_threads=num_cpu_threads,
                   batch_size=batch_size,
                   file_root=file_root,
                   sampler=sampler,
                   sample_rate=sample_rate,
                   resample_range=resample_range,
                   window_size=window_size,
                   window_stride=window_stride,
                   nfeatures=nfeatures,
                   nfft=nfft,
                   dither_coeff=dither_coeff,
                   silence_threshold=silence_threshold,
                   preemph_coeff=preemph_coeff,
                   max_duration=max_duration,
        )

    @staticmethod
    def _rocal_init_log(args: dict):
        if (not torch.distributed.is_initialized() or (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):  # print once
            max_len = max([len(ii) for ii in args.keys()])
            fmt_string = '\t%' + str(max_len) + 's : %s'
            print('Initializing ROCAL with parameters:')
            for keyPair in sorted(args.items()):
                print(fmt_string % keyPair)
    
    def RocalPipline(self):
        _rocal_cpu = True #TODO : Introduce a param from user for cpu / gpu backend
        audio_pipeline = Pipeline(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id, rocal_cpu=_rocal_cpu)
        with audio_pipeline:
            jpegs, labels = fn.readers.file(file_root=self.file_root_rocal, file_list=self.sampler.get_file_list_path())
            speed_perturbation_coeffs = fn.uniform(rng_range=[0.85, 1.15])
            sample_rate = 16000
            # Get the correct dataset path instead of hardcoding the dataset.
            audio_decode = fn.decoders.audio(jpegs, file_root=self.file_root_rocal, sample_rate=self.sample_rate )
            begin_and_length = fn.nonsilent_region(audio_decode) # Dont understand where to use this as input in Slice to pass as what arguments - Confused
            trim_silence = fn.slice(audio_decode, normalized_anchor=False, normalized_shape=False, axes=[0], anchor=[0], shape=[4], fill_values=[0.3])
            preemph_audio = fn.preemphasis_filter(trim_silence, preemph_coeff=self.preemph_coeff)
            spectogram = fn.spectrogram(preemph_audio, nfft=self.nfft, window_length=int(self.window_size * self.sample_rate), window_step= int(self.window_stride * self.sample_rate))
            mel_fbank = fn.mel_filter_bank(spectogram, sample_rate=self.sample_rate, nfilter=self.nfeatures, normalize=True)
            to_decibels = fn.to_decibals(mel_fbank, rocal_tensor_output_type=types.FLOAT)
            normalize = fn.normalize(to_decibels, axes=[1])
            padded_audio = fn.pad(normalize, fill_value=0)
            audio_pipeline.set_outputs(padded_audio)

        audio_pipeline.build()
        return audio_pipeline


