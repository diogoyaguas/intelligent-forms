"""NLP Pipeline Module."""

from typing import Any, List, Union


class Pipeline:
    """Pipeline for process text."""

    def __init__(self, preprocessing: List[Any]) -> None:
        """Pipeline for process text.

        Args:
            preprocessing (List[Any]): List of callable objects with implemented method ```forward```.
        """
        self.preprocessing = preprocessing

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Process Text.

        Args:
            text (Union[str, List[str]]): Text to be processed.

        Returns: Union[str, List[str]]
        """
        for t in self.preprocessing:
            text = t.forward(text)
        return text
