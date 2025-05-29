import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Union, Optional

class TimeSeriesDataset(Dataset):
    """Time Series Dataset for PyTorch
    
    This dataset class is designed for time series data, supporting both single-step and multi-step predictions.
    It handles the creation of input sequences and corresponding target values for time series forecasting.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input time series data. If DataFrame, index should be datetime.
    features : List[str]
        List of feature column names to use as input
    target : str
        Target column name to predict
    sequence_length : int
        Length of input sequences (lookback window)
    prediction_length : int, optional
        Length of prediction horizon, default=1
    stride : int, optional
        Step size between sequences, default=1
    transform : callable, optional
        Optional transform to be applied on the data
    target_transform : callable, optional
        Optional transform to be applied on the target
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        features: List[str],
        target: str,
        sequence_length: int,
        prediction_length: int = 1,
        stride: int = 1,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform
        
        # Convert numpy array to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            data = pd.DataFrame(data)
            
        # Store features and target
        self.features = features
        self.target = target
        
        # Extract feature and target data
        self.X = data[features].values
        self.y = data[target].values
        
        # Calculate number of samples
        self.n_samples = (len(data) - sequence_length - prediction_length + 1) // stride
        
        # Store original index if available
        self.index = data.index if isinstance(data, pd.DataFrame) else None
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset
        
        Parameters
        ----------
        idx : int
            Index of the sample to get
            
        Returns
        -------
        tuple
            (X, y) where X is the input sequence and y is the target
        """
        # Calculate start and end indices for the sequence
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Get input sequence
        X = self.X[start_idx:end_idx]
        
        # Get target values
        y = self.y[end_idx:end_idx + self.prediction_length]
        
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Apply transforms if specified
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
            
        return X, y
    
    def get_sequence_indices(self, idx: int) -> tuple:
        """Get the start and end indices for a given sample index
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        tuple
            (start_idx, end_idx) for the sequence
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        return start_idx, end_idx
    
    def get_original_index(self, idx: int) -> Union[pd.DatetimeIndex, None]:
        """Get the original datetime index for a given sample index
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        Union[pd.DatetimeIndex, None]
            Original datetime index if available, None otherwise
        """
        if self.index is not None:
            start_idx, end_idx = self.get_sequence_indices(idx)
            return self.index[start_idx:end_idx]
        return None
    
    @property
    def input_shape(self) -> tuple:
        """Get the shape of input sequences
        
        Returns
        -------
        tuple
            (sequence_length, n_features)
        """
        return (self.sequence_length, len(self.features))
    
    @property
    def output_shape(self) -> tuple:
        """Get the shape of output sequences
        
        Returns
        -------
        tuple
            (prediction_length,)
        """
        return (self.prediction_length,)
    
    def __repr__(self) -> str:
        """String representation of the dataset"""
        return (f"TimeSeriesDataset(n_samples={self.n_samples}, "
                f"sequence_length={self.sequence_length}, "
                f"prediction_length={self.prediction_length}, "
                f"n_features={len(self.features)})")

def create_time_series_dataset(
    data: Union[pd.DataFrame, np.ndarray],
    features: List[str],
    target: str,
    sequence_length: int,
    prediction_length: int = 1,
    stride: int = 1,
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None
) -> TimeSeriesDataset:
    """Helper function to create a TimeSeriesDataset
    
    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input time series data
    features : List[str]
        List of feature column names
    target : str
        Target column name
    sequence_length : int
        Length of input sequences
    prediction_length : int, optional
        Length of prediction horizon, default=1
    stride : int, optional
        Step size between sequences, default=1
    transform : callable, optional
        Optional transform to be applied on the data
    target_transform : callable, optional
        Optional transform to be applied on the target
        
    Returns
    -------
    TimeSeriesDataset
        Created dataset
    """
    return TimeSeriesDataset(
        data=data,
        features=features,
        target=target,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        stride=stride,
        transform=transform,
        target_transform=target_transform
    )

if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    }, index=dates)
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data=data,
        features=['feature1', 'feature2'],
        target='target',
        sequence_length=10,
        prediction_length=1
    )
    
    # Get a sample
    X, y = dataset[0]
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Get original indices
    indices = dataset.get_original_index(0)
    print(f"Original indices: {indices}") 