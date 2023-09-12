"""
see https://cloud.ifremer.fr/index.php/s/ejiddQDvvHS2n0I and https://trello.com/c/MHbwN59w/583-generate-d1v4
"""
import numpy as np
import h5py
import math
from tensorflow.keras.utils import Sequence
import logging
class SARGenerator(Sequence):
    """
    Generator to be passed to fit_generator, predict_generator, etc. using Keras interface.
    
    Arguments:
    filename (string) -- path to h5py file containing data
    groups (string) -- A list of hdf5 group names. E.G. ['2015_2016', '2017'] 
                       If hdf5 file has no group structure, will default to root.
    batch_size (int) -- number of data points to read out at a time. (last read may have less than batch size)
    """
    def __init__(self, filename, batch_size=100, subgroups=None):
        """

        :param filename:
        :param batch_size:
        :param subgroups:
        """
        self.spectra_varname = 'spectrum_slc'
        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.batch_size = batch_size
        # In case of no subgroups, use special subgroup with key=None.
        self.groups = subgroups if subgroups is not None else [None]
        self._calc_batches_per_group()



    def __del__(self):
        self.h5file.close()
        
    def _num_examples_group(self, group):
        """Number of notebooks in file."""

        if group is None:
            return self.h5file[self.spectra_varname].shape[0]
        else:
            return self.h5file[group][self.spectra_varname].shape[0]
    
    def _calc_batches_per_group(self):
        """
        Calc num batches per group. Also an index of which batch comes from which group.
        """
        batches_per_group = {}
        for group in self.groups:
            num_group = self._num_examples_group(group)
            batches_per_group[group] = math.ceil(num_group / self.batch_size)
        self.batches_per_group = batches_per_group
        
        # Total number of batches.
        num_batches_total = 0
        for g, num_batches in batches_per_group.items():
            num_batches_total += num_batches
        self.num_batches_total = num_batches_total
        
        # 
        self.idx2igroup = np.zeros((num_batches_total, ), dtype=int)
        self.idx2inbatchidx = np.zeros((num_batches_total, ), dtype=int) # Within-group index.
        self.igroup2group = {}
        count = 0
        for i, (g, n) in enumerate(batches_per_group.items()):
            self.igroup2group[i] = g
            self.idx2igroup[count:count+n] = i
            self.idx2inbatchidx[count:count+n] = np.arange(0, n, dtype=int)
            #logging.info('correction apllied')
            count += n #correction to use also 2017 agrouaze June 2021
        return
    
    def __len__(self):
        """Number of batches. Not all batches must be full."""
        return self.num_batches_total

    def __getitem__(self, idx):
        """Return batch."""
        return self._get_batch_contiguous(idx)
    
    def _get_batch_contiguous(self, idx):
        """Return batch contiguous. This will be faster, but hard to shuffle data."""
        #logging.info('start contigous')
        group = self.igroup2group[self.idx2igroup[idx]]
        start = self.batch_size * self.idx2inbatchidx[idx]
        stop = np.minimum(start + self.batch_size, self._num_examples_group(group))
        dataset = self.h5file if group is None else self.h5file[group]
        #logging.info('dataset : %s',dataset)
        # Image spectra.
        spectrum = dataset[self.spectra_varname][start:stop]
        assert spectrum.shape[1:] == (72, 60, 2)
        assert not np.any(np.isnan(spectrum))
        #assert not np.any(spectrum > 10000), spectrum.max()
        #logging.info('set zero  high level spectrum : ')
        spectrum[spectrum > 1000000] = 0 # increased by agrouaze 1e6 (max observed on a given day after normalization 99 percentil)
        
        # High level features. Should be preprocessed already.

        names = [ 'latlonSARcossin', 'doySAR', 'incidence'] # 'wsALT' pas present le 15oct mais il faut que je le mette
        features = []
        for name in names:
            #logging.info('name in gen : %s',name)
            if name in dataset:

                temp = dataset[name][start:stop]
                if len(temp.shape)==1:
                    temp = np.reshape(temp,(len(temp),1))
                #logging.info('%s %s',name,temp.shape)
            else:
                raise Exception('%s is not present in the input training dataset'%name)
            features.append(temp)
        features = np.hstack(features)
        #logging.info('generator high level feature : %s',features)
        #features = np.hstack([self.data[name][start:stop] for name in names])
        assert features.shape[1] == 6, features.shape
        #logging.info('assertion shape ok')
        assert not np.any(np.isnan(features))
        assert not np.any(features > 1000), features.max()
        #logging.info('assertion ok')
        # Target in m. 
        if 'hsALT' in dataset:
            target = dataset['hsALT'][start:stop]
            assert target.shape[1] == 1
            assert not np.any(np.isnan(target))
            assert not np.any(target > 100), target
        else:
            target = None
        #logging.info('spectrum : %s',spectrum.shape)
        inputs = [spectrum, features]
        outputs = target
        return inputs, outputs