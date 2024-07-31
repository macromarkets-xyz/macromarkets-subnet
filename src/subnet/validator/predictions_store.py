from cachetools import LRUCache
from typing import Dict

from common.prediction import Prediction


class PredictionCache:
    def __init__(self, maxsize: int = 256):
        self.cache: LRUCache[str, Dict[str, Prediction]] = LRUCache(maxsize=maxsize)

    def get(self, key: str) -> Dict[str, Prediction] | None:
        return self.cache.get(key)

    def set(self, key: str, value: Dict[str, Prediction]) -> None:
        self.cache[key] = value

    def remove(self, key: str) -> None:
        self.cache.pop(key, None)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        return key in self.cache


# Usage example
if __name__ == "__main__":
    cache = PredictionCache(maxsize=256)

    # Adding items to the cache
    cache.set("key1", {"pred1": Prediction(), "pred2": Prediction()})
    cache.set("key2", {"pred3": Prediction(), "pred4": Prediction()})

    # Retrieving items from the cache
    print(cache.get("key1"))  # Will return the dict of Predictions
    print(cache.get("key3"))  # Will return None

    # Checking if a key exists
    print("key2" in cache)  # Will return True

    # Getting the number of items in the cache
    print(len(cache))  # Will return 2

    # Removing an item
    cache.remove("key1")

    # Clearing the cache
    cache.clear()
