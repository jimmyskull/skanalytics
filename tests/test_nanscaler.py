import unittest

import pandas as pd
import numpy as np

from .context import nanscale, StandardNANScaler


class TestNANScaler(unittest.TestCase):

    def setUp(self):
        # Without NaN
        self.np_without_nans = np.random.random_sample((100, 4))
        self.df_without_nans = pd.DataFrame(self.np_without_nans)
        # With NaN
        self.np_with_nans = np.copy(self.np_without_nans)
        indices = list(np.ndindex(self.np_with_nans.shape))
        np.random.shuffle(indices)
        self.nan_count = 100
        self.np_with_nans[indices[:self.nan_count]] = np.nan
        self.df_with_nans = pd.DataFrame(self.np_with_nans)

    def test_class_numpy_without_nans(self):
        data = self.np_without_nans
        shape_before = np.array(data.shape)
        self.assertTrue(np.count_nonzero(~np.isnan(data))== 0)
        scaler = StandardNANScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        self.assertTrue(np.all(shape_before == np.array(result.shape)))
        self.assertTrue(np.count_nonzero(~np.isnan(result))== 0)

    def test_class_pandas_without_nans(self):
        data = self.df_without_nans
        shape_before = np.array(data.shape)
        self.assertTrue(np.count_nonzero(~np.isnan(data))== 0)
        scaler = StandardNANScaler()
        scaler.fit(data)
        result = pd.DataFrame(scaler.transform(data), columns=data.columns)
        self.assertTrue(np.all(shape_before == np.array(result.shape)))
        self.assertTrue((df.isnull() == data.isnull()).all().all())
        self.assertTrue(np.allclose(df.mean(), 0))
        self.assertTrue(df.isnull().sum().sum() == 0)

    def test_sklearn_object_with_nans(self):
        data = self.data_with_nans
        self.assertTrue(data.isnull().sum().sum() == self.nan_count)
        scaler = StandardNANScaler()
        scaler.fit(self.data)
        df = pd.DataFrame(scaler.transform(self.data),
                          columns=self.data.columns)
        self.assertTrue((df.isnull() == self.data.isnull()).all().all())
        self.assertTrue(np.allclose(df.mean(), 0))
        self.assertTrue(df.isnull().sum().sum() == self.nan_count)

    def test_standalone_function_without_nans(self):
        data = self.data_without_nans
        self.assertTrue(data.isnull().sum().sum() == 0)
        df = pd.DataFrame(nanscale(self.data), columns=self.data.columns)
        self.assertTrue((df.isnull() == self.data.isnull()).all().all())
        self.assertTrue(np.allclose(df.mean(), 0))
        self.assertTrue(df.isnull().sum().sum() == 0)

    def test_standalone_function_with_nans(self):
        data = self.data_with_nans
        self.assertTrue(data.isnull().sum().sum() == 0)
        df = pd.DataFrame(nanscale(data), columns=data.columns)
        self.assertTrue((df.isnull() == data.isnull()).all().all())
        self.assertTrue(np.allclose(df.mean(), 0))
        self.assertTrue(df.isnull().sum().sum() == self.nan_count)

if __name__ == '__main__':
    unittest.main()
