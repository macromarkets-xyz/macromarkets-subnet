import math

from pydantic import BaseModel
from typing import List, Optional
from statistics import mean
import sys


class MinerStats(BaseModel):
    accuracy_scores: List[float] = []
    last_block: int = sys.maxsize

    def avg_accuracy(self) -> float:
        if not self.accuracy_scores:
            return 0.0
        return mean(self.accuracy_scores)

    def __str__(self) -> str:
        return f"MinerStats(accuracy_scores={self.accuracy_scores}, last_block={self.last_block}, avg_accuracy={self.avg_accuracy()})"
