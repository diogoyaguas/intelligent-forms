"""IASAE Complaint Module."""

import os
import os.path
import glob
from os import path
from typing import Any, List, Tuple, Callable

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

from nlp.utils import build_vocab_from_iterator
import classifiers.utils
import shutil
import pickle
from django.db import connection
from classifiers.utils import train_test_split

tqdm.pandas()


class TextClassificationDataset(torch.utils.data.Dataset):
    """Abstract Text Classification class."""

    def __init__(self, data: pd.DataFrame, labels: List[str]) -> None:
        """Abstract Text Classification class.

        Args:
            data (pd.DataFrame): dataframe with the dataset data.
            labels (List[str]): list of labels.
        """
        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels

    def __getitem__(self, i: int) -> Tuple[str, int]:
        """Get item from a dataframe.

        Args:
            i (int): index of the sample

        Returns: Tuple[str, int]
        """
        return self._data.iloc[i].to_list()

    def __len__(self) -> int:
        """Get Dataset length.

        Returns: int
        """
        return len(self._data)

    def __iter__(self) -> Any:
        """Iterate Dataset.

        Returns: Any

        """
        for _, x in self._data.content.iteritems():
            yield x

    def get_labels(self) -> List[str]:
        """Get dataset labels.

        Returns: List[str]
        """
        return self._labels


def COMPLAINTS(pickle_path: str, task: str, output: str, pipeline: Callable = None, classes: dict = None, save: bool = False
               ) -> Tuple[TextClassificationDataset, TextClassificationDataset, TextClassificationDataset]:
    """Generate train, validation and test dataset.

    Args:
        pickle_path (str): path to pickle dataframe.
        task (str): chose training task.
        pipeline (Callable): Callable object.
        save (bool): if true will save a pickle file with name dataset.pickle

    Returns: Tuple[TextClassificationDataset, TextClassificationDataset, TextClassificationDataset]

    """    
    print(classes)
    
    from complaints.models import Complaint
    query = str(Complaint.objects.all().query)
    df = pd.read_sql_query(query, connection)
    df = train_test_split(df, task, classes, 0.2)
    out=f'{output}'

    if task == "economic_class":
        df = df[df.economic_class != "Não Existe"]

    if(not path.exists(f"{out}/dataset.pickle")):
        os.makedirs(f"{out}/tmp")
        if pipeline:
            df = df.reset_index(drop=True) 
            df1 = pd.DataFrame()
            df1 = df.loc[0:1000]

            for idx, row in tqdm(df.iterrows()):
                df1.at[idx, "content"] = pipeline(row.content)
                if idx % 1000 == 0 and save:
                    df1.to_pickle(f"{out}/tmp/dataset_{str(idx)}.pickle")
                    df1 = pd.DataFrame()
                    df1 = df.loc[idx:idx+1000]
            if save:
                files = glob.glob(f'{out}/tmp/*.pickle')
                df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
                df.to_pickle(f"{out}/dataset.pickle")
                shutil.rmtree(f'{out}/tmp')
    else: 
        df = pd.read_pickle(f"{out}/dataset.pickle")
        df = df[df.economic_class != "Não Existe"]

    if task != "infraction_class":
        le = LabelEncoder()
        df["labels"] = pd.DataFrame({'labels': list(le.fit_transform(df[task]))})
        labels = le.classes_
        pickle.dump(le, open(f"{out}/model/label_vectorizer.pkl", 'wb'))
    else:
        multilabel = MultiLabelBinarizer()
        df["labels"] = pd.DataFrame({'labels': list(multilabel.fit_transform(df[task]))})
        labels = multilabel.classes_
        pickle.dump(multilabel, open(f"{out}/model/label_vectorizer.pkl", 'wb'))

    df_test = df[~df[f"{task}_train"]].reset_index(drop=True)
    df_val = df[df[f"{task}_validation"]].reset_index(drop=True)
    df = df[df[f"{task}_train"] & ~df[f"{task}_validation"]].reset_index(drop=True)
    
    columns = ["content", "labels"]
    return (TextClassificationDataset(df[columns], labels),         # Train dataset
            TextClassificationDataset(df_val[columns], labels),     # Validation dataset
            TextClassificationDataset(df_test[columns], labels))    # Test dataset


class ComplaintAsaeDataModule(pl.LightningDataModule):
    """IASAE Complaint Data Module."""

    def __init__(self, pickle_path: str, task: str, output: str, generate_batch: Callable = None, pipeline: Callable = None,
                 classes: dict = None, batch_size: int = 16, save: bool = False) -> None:
        """Complaint ASAE Dataset initialization.

        Args:
            pickle_path (str): path to pickle dataframe.
            task (str): chose training task.
            generate_batch (callable): callable to process the text documents.
            pipeline (Callable): Callable object.
            batch_size (int): training batch size.
            save (bool): if true will save a pickle file with name dataset.pickle
        """
        super().__init__()
        self.pickle_path = pickle_path
        self.task = task
        self.output = output
        self.pipeline = pipeline
        self.classes = classes
        self.generate_batch = generate_batch
        self.batch_size = batch_size
        self.num_workers = os.cpu_count()# or 2
        self.save = save

    def setup(self, stage=None) -> None:
        """Get Data Module setup for training."""
        self.complaint_train, self.complaint_val, self.complaint_test = COMPLAINTS(self.pickle_path,
                                                                                   self.task,
                                                                                   self.output,
                                                                                   self.pipeline,
                                                                                   self.classes,
                                                                                   self.save)
        self.labels = self.complaint_train.get_labels()
        self.vocab = build_vocab_from_iterator(self.complaint_train)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(self.complaint_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        """Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(self.complaint_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        """Create Dataloader.

        Returns: DataLoader
        """
        return DataLoader(self.complaint_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)
