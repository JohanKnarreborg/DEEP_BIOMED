from monai.data import CacheDataset

class RepeatedCacheDataset(CacheDataset):
    """
    The code of this function was developed from following course 02510 
    Deep Learning for Experimental 3D Image Analysis at DTU.
    See course website here: https://www2.compute.dtu.dk/courses/02510/
    
    CacheDataset that repeats the data.
    """

    def __init__(self, *args, num_repeats: int = 1, **kwargs):
        """
        Args:
            *args: Arguments to pass to CacheDataset.
            num_repeats (int): Number of times to repeat the data.
            **kwargs: Keyword arguments to pass to CacheDataset.
        """
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats


    def __len__(self):
        """Returns number of items in the dataset."""
        return super().__len__() * self.num_repeats


    def __getitem__(self, index):
        """Returns the item at the given index."""
        index = index % super().__len__()
        return super().__getitem__(index)