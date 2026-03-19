"""Leave-One-Subject-Out (LOSO) Data Splitter.

Ensures rigorous cross-dataset isolation by enforcing structural disjointness 
between training subsets and validation subsets per fold.
"""
import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from torch.utils.data import Subset, DataLoader, Dataset


class LOSOCrossValidator:
    """LOSO Generator producing sequentially isolated training geometries."""

    def __init__(self, metadata_df: pd.DataFrame, base_dataset: Dataset) -> None:
        """
        Parameters
        ----------
        metadata_df : pd.DataFrame
            DataFrame holding structured columns, critically including 'subject'.
            Must systematically match the exact positional index progression of `base_dataset`.
        base_dataset : Dataset
            The underlying PyTorch Dataset encompassing the entire composite database.
        """
        if "subject" not in metadata_df.columns:
            raise KeyError("The provided parameter `metadata_df` critically requires a 'subject' tracking column.")
            
        if len(metadata_df) != len(base_dataset):
            raise ValueError(
                f"Dimensional mismatch: metadata matrix length ({len(metadata_df)}) "
                f"vs underlying base dataset size ({len(base_dataset)})."
            )
            
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.base_dataset = base_dataset
        
        # Extract the underlying distinct identities
        self.subjects = self.metadata_df['subject'].unique()

    def get_folds(
        self, batch_size: int, num_workers: int = 4
    ) -> Iterator[Tuple[int, str, DataLoader, DataLoader]]:
        """A generator emitting disjoint permutations.
        
        Parameters
        ----------
        batch_size : int
            Uniform allocation size for the yielded DataLoaders.
        num_workers : int
            Multiprocessing background workers for pipelined augmentation routines.
            
        Yields
        ------
        fold_idx : int
            Progress enumerator zero-index integer.
        test_subject_id : str or int
            The explicit identity isolated and withheld from the training subset.
        train_loader : DataLoader
            Loader pointing to indices mathematically strictly avoiding `test_subject_id` (Shuffled).
        test_loader : DataLoader
            Loader representing data uniformly tracking to `test_subject_id` (Sequential).
        """
        for fold_idx, test_subject in enumerate(self.subjects):
            # Formulate strict boolean vector matrices
            test_mask = self.metadata_df['subject'] == test_subject
            train_mask = ~test_mask
            
            # Map valid subset configurations extracting zero-indexed positions
            test_indices = [int(x) for x in np.where(test_mask)[0]]
            train_indices = [int(x) for x in np.where(train_mask)[0]]
            
            # Establish isolated abstract geometries utilizing zero sequence memory leakage
            train_subset = Subset(self.base_dataset, train_indices)
            test_subset = Subset(self.base_dataset, test_indices)
            
            # Formulate Loaders ensuring sequential structure mapping properties
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,  # Stochastic training allocation
                num_workers=num_workers,
                pin_memory=True,
                # Architecture uses LayerNorm instead of BatchNorm, no need to drop last batch.
                # Dropping data in a small, highly imbalanced dataset like CAS(ME)² deletes critical minority-class examples.
                drop_last=False
            )
            
            test_loader = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False, # Maintain contiguous deterministic inference sequence
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
            
            yield fold_idx, test_subject, train_loader, test_loader
