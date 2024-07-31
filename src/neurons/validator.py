# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
import datetime

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
import datetime as dt
import os
import math
import time
import random
import shutil
import asyncio
import subprocess
import argparse
import typing

import numpy as np
import requests
from importlib.metadata import version
from shlex import split
from scipy import optimize, stats

import multiprocessing
from rich.table import Table
from rich.console import Console
import traceback
import bittensor as bt

import torch
import constants
from common.assets import ASSET_IDS

from common.mainlog import MainLogger
from common.prediction import Prediction

from subnet.validator.miner_stats import MinerStats
from subnet.validator.predictions_store import PredictionCache

main_logger = MainLogger()


@dataclass
class LocalMetadata:
    """Metadata associated with the local validator instance"""

    commit: str
    btversion: str
    uid: int = 0
    coldkey: str = ""
    hotkey: str = ""


def local_metadata() -> LocalMetadata:
    """Extract the version as current git commit hash"""
    commit_hash = ""
    try:
        result = subprocess.run(
            split("git rev-parse HEAD"),
            check=True,
            capture_output=True,
            cwd=constants.ROOT_DIR,
        )
        commit = result.stdout.decode().strip()
        assert len(commit) == 40, f"Invalid commit hash: {commit}"
        commit_hash = commit[:8]
    except:
        commit_hash = "unkown"

    bittensor_version = version("bittensor")
    return LocalMetadata(
        commit=commit_hash,
        btversion=bittensor_version,
    )


def accuracy_score(prediction: Optional[float], actual: Optional[float]) -> float:
    """
    Calculate the accuracy score as a percentage.


    :param prediction: The predicted or quoted value
    :param actual: The actual or true value
    :return: Accuracy score as a percentage (0 to 100)
    """
    # Skip when there is no price available
    if actual is None:
        return 0

    if prediction is None:
        return 0

    relative_error = abs((prediction - actual) / actual)
    accuracy = (1 - relative_error) * 100

    # Clamp the accuracy between 0 and 100
    return max(0.0, min(100.0, accuracy))


def exit_early():
    return False


# fetches time from last 24 hours ago
def get_last_time_period():
    return 5


class Validator:
    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        # Default 100 blocks ~= 20 minutes
        parser.add_argument(
            "--blocks_per_epoch",
            type=int,
            # default=100,
            default=10,
            help="Number of blocks to wait before setting weights.",
        )
        parser.add_argument(
            "--offline",
            action="store_true",
            help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
        )
        parser.add_argument(
            "--netuid", type=str, default=constants.SUBNET_UID, help="The subnet UID."
        )
        parser.add_argument(
            "--wandb-key",
            type=str,
            default="",
            help="A WandB API key for logging purposes",
        )
        bt.logging.off()
        bt.subtensor.add_args(parser)
        bt.wallet.add_args(parser)

        bt.axon.add_args(parser)
        config = bt.config(parser)
        return config

    def __init__(self, local_metadata: LocalMetadata):
        self.config = Validator.config()
        main_logger = MainLogger()

        main_logger.info(f"valiator_init", config=self.config)

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(self.config.netuid)

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.last_epoch = self.metagraph.block.item()
        validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        if validator_uid is None and not self.config.offline:
            raise ValueError(f"You are not registered on netuid {self.config.netuid}")

        # Set up local metadata for stats collection
        self.local_metadata = LocalMetadata(
            commit=local_metadata.commit,
            btversion=local_metadata.btversion,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            uid=validator_uid,
        )
        self.penalty = 0.03
        # Setup timeseries api
        self.timeseries_api_url = self.config.tapi or "localhost:9999"

        main_logger.info(
            "validator_setup",
            local_metadata=self.local_metadata,
            last_epoch=self.last_epoch,
            len_weights=len(self.weights),
            validator_uid=validator_uid,
        )

    def __del__(self):
        if hasattr(self, "stop_event"):
            self.stop_event.set()

    def fetch_prices(self, time: str):
        return {"C=F": 5}

    async def try_set_weights(self, ttl: int):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num(0.0)
                self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=self.weights,
                    wait_for_inclusion=False,
                    version_key=constants.weights_version_key,
                )
                weights_report = {"weights": {}}
                for uid, score in enumerate(self.weights):
                    weights_report["weights"][uid] = score

            except Exception as e:
                main_logger.error(f"failed to set weights {e}")
            ws, ui = self.weights.topk(len(self.weights))
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)

        try:
            bt.logging.debug("Setting weights.")
            await asyncio.wait_for(_try_set_weights(), ttl)
            bt.logging.debug("Finished setting weights.")
        except asyncio.TimeoutError:
            bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int) -> bool:
        def sync_metagraph(queue: multiprocessing.Queue):
            try:
                # Update self.metagraph
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                self.metagraph.save()
            except Exception as e:
                queue.put(e)
            else:
                queue.put(None)

        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=sync_metagraph, args=(queue,))
        process.start()
        process.join(timeout=ttl)

        if process.is_alive():
            process.terminate()
            process.join()
            main_logger.error(f"metagraph_sync_timeout", timeout=ttl)
            return False

        try:
            exception: Optional[Exception] = queue.get(block=False)
            if exception is not None:
                main_logger.error(f"metagraph_sync_error", error=str(exception))
                return False
        except Exception as e:
            main_logger.error("metagraph_sync_unknown_error", error=e)
            return False

        main_logger.info("metagraph_sync_success")
        self.metagraph.load()
        # self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
        return True

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()

        try:
            await asyncio.wait_for(_try_run_step(), ttl)
            main_logger.warning("step_complete")
        except asyncio.TimeoutError:
            main_logger.error(f"step_timeout", timeout=ttl)
            return
        except Exception as e:
            main_logger.error(f"step_fail", error=e)

    def queue_models(self):
        uids = self.metagraph.uids.tolist()
        # Prevent uid order bias
        random.shuffle(uids)
        for uid in uids:
            hotkey = self.metagraph.hotkeys[uid]
            metadata = bt.extrinsics.serving.get_metadata(
                self.subtensor, self.config.netuid, hotkey
            )
            if metadata is None:
                continue
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            chain_str = bytes.fromhex(hex_data).decode()
            block = metadata["block"]

    def fetch_actual_price(self, asset_id: str, time: str) -> Optional[float]:
        return random.random()

    def get_miner_prediction(
        self, hotkey: str, asset_id: str, time_period: str
    ) -> Optional[float]:
        try:
            request_payload = {
                "hotkey": hotkey,
                "asset_id": asset_id,
                "time_period": time_period,
            }
            # Fetch computed prediction from timeseries api
            response = requests.post(self.timeseries_api_url, json=request_payload)
            if response.status_code != 200:
                main_logger.info("response_success", text=response.text)
            response_data = response.json()
            prediction = response_data.get("prediction", None)
            return prediction

        except Exception as e:
            main_logger.error("prediction_fetch_fail", error=e)
            return random.random()
            # return None

    def compute_accuracy_scores(
        self,
        miner_stats: Dict[int, MinerStats],
        asset_id: str,
        time: str,
    ) -> Dict[int, MinerStats]:
        actual_price = self.fetch_actual_price(asset_id=asset_id, time=time)
        for uid in miner_stats.keys():
            miner_hotkey = self.metagraph.hotkeys[uid]
            prediction = self.get_miner_prediction(
                hotkey=miner_hotkey, asset_id="", time_period=time
            )
            accuracy = accuracy_score(actual=actual_price, prediction=prediction)
            miner_stats[uid].accuracy_scores.append(accuracy)
        return miner_stats

    def iswin(self, score_i, score_j, block_i, block_j):
        """
        Determines the winner between two models based on the epsilon adjusted loss.

        Parameters:
            loss_i (float): Loss of uid i on batch
            loss_j (float): Loss of uid j on batch.
            block_i (int): Block of uid i.
            block_j (int): Block of uid j.
        Returns:
            bool: True if loss i is better, False otherwise.
        """
        # Adjust score based on timestamp and pretrain epsilon
        score_i = (1 - self.penalty) * score_i if block_i > block_j else score_i
        score_j = (1 - self.penalty) * score_j if block_j > block_i else score_j
        return score_i > score_j

    def compute_wins(self, miner_stats: Dict[int, MinerStats]):
        uids = miner_stats.keys()
        wins = {uid: 0 for uid in uids}
        win_rate = {uid: 0 for uid in uids}
        for i, uid_i in enumerate(uids):
            total_matches = 0
            block_i = miner_stats[uid_i].last_block
            for j, uid_j in enumerate(uids):
                if i == j:
                    continue
                block_j = miner_stats[uid_j].last_block
                score_i = miner_stats[uid_i].avg_accuracy()
                score_j = miner_stats[uid_j].avg_accuracy()
                wins[uid_i] += (
                    1 if self.iswin(score_i, score_j, block_i, block_j) else 0
                )
                total_matches += 1
            # Calculate win rate for uid i
            win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

        return wins, win_rate

    async def run_step(self):
        """
        Executes a step in the evaluation process of models. This function performs several key tasks:
        """
        last_time_period = get_last_time_period()
        # Update self.metagraph
        synced = await self.try_sync_metagraph(ttl=60)
        if not synced:
            return
        # assume latest metagraph is synced
        uids = self.metagraph.uids.tolist()
        main_logger.info("sync_status", synced=synced, uids=uids)
        miner_stats: dict[int, MinerStats] = {uid: MinerStats() for uid in uids}
        for asset_id in ASSET_IDS:
            miner_stats = self.compute_accuracy_scores(
                miner_stats=miner_stats, asset_id=asset_id, time=str(last_time_period)
            )
        forlog = ""
        for k, v in miner_stats.items():
            forlog = f"{forlog} {k}={str(v)}"
        main_logger.info("scores_calculated", stats=str(forlog))
        wins, win_rate = self.compute_wins(miner_stats=miner_stats)

        # # Compute softmaxed weights based on win rate.
        model_weights = torch.tensor(
            [win_rate[uid] for uid in uids], dtype=torch.float32
        )
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        # Update weights based on moving average.
        new_weights = torch.zeros_like(self.metagraph.S)
        for i, uid_i in enumerate(uids):
            new_weights[uid_i] = step_weights[i]
        scale = 1
        new_weights *= scale / new_weights.sum()
        if new_weights.shape[0] < self.weights.shape[0]:
            self.weights = self.weights[: new_weights.shape[0]]
        elif new_weights.shape[0] > self.weights.shape[0]:
            self.weights = torch.cat(
                [
                    self.weights,
                    torch.zeros(new_weights.shape[0] - self.weights.shape[0]),
                ]
            )
        self.weights = (
            constants.alpha * self.weights + (1 - constants.alpha) * new_weights
        )
        self.weights = self.weights.nan_to_num(0.0)
        concensus = self.metagraph.C.cpu().numpy()
        weights_np = self.weights.cpu().numpy()
        self.weights = self.adjust_for_vtrust(weights_np, consensus=concensus)
        main_logger.info("weights_calculated", weights=self.weights)
        # self.try_set_weights()

    async def run(self):
        while True:
            try:
                # Time based weight setting
                next_time = next_interval()
                # next_time_sleep = (next_time - datetime.datetime.now(dt.timezone.utc)).total_seconds()
                next_time_sleep = 10
                main_logger.info("sleeping", next_time_sleep=next_time_sleep)
                time.sleep(next_time_sleep)
                await self.try_run_step(ttl=60 * 20)
                self.last_epoch = self.metagraph.block.item()
                time.sleep(1)

            except KeyboardInterrupt:
                main_logger.info("KeyboardInterrupt caught")
                exit()
            except Exception as e:
                main_logger.error(
                    f"Error in validator loop \n {e} \n {traceback.format_exc()}"
                )

    def adjust_for_vtrust(self, weights, consensus, vtrust_min: float = 0.5):
        """
        Interpolate between the current weight and the normalized consensus weights so that the
        vtrust does not fall below vturst_min, assuming the consensus does not change.
        """
        vtrust_loss_desired = 1 - vtrust_min

        # If the predicted vtrust is already above vtrust_min, then just return the current weights.
        orig_vtrust_loss = np.maximum(0.0, weights - consensus).sum()
        if orig_vtrust_loss <= vtrust_loss_desired:
            main_logger.info(
                "Weights already satisfy vtrust_min. {} >= {}.".format(
                    1 - orig_vtrust_loss, vtrust_min
                )
            )
            return weights

        # If maximum vtrust allowable by the current consensus is less that vtrust_min, then choose the smallest lambda
        # that still maximizes the predicted vtrust. Otherwise, find lambda that achieves vtrust_min.
        vtrust_loss_min = 1 - np.sum(consensus)
        if vtrust_loss_min > vtrust_loss_desired:
            main_logger.info(
                "Maximum possible vtrust with current consensus is less than vtrust_min. {} < {}.".format(
                    1 - vtrust_loss_min, vtrust_min
                )
            )
            vtrust_loss_desired = 1.05 * vtrust_loss_min

        # We could solve this with a LP, but just do rootfinding with scipy.
        consensus_normalized = consensus / np.sum(consensus)

        def fn(lam: float):
            new_weights = (1 - lam) * weights + lam * consensus_normalized
            vtrust_loss = np.maximum(0.0, new_weights - consensus).sum()
            return vtrust_loss - vtrust_loss_desired

        sol = optimize.root_scalar(fn, bracket=[0, 1], method="brentq")
        lam_opt = sol.root

        new_weights = (1 - lam_opt) * weights + lam_opt * consensus_normalized
        vtrust_pred = np.minimum(weights, consensus).sum()
        main_logger.info(
            "Interpolated weights to satisfy vtrust_min. {} -> {}.".format(
                1 - orig_vtrust_loss, vtrust_pred
            )
        )
        return new_weights


def next_interval():
    now = datetime.datetime.now(dt.timezone.utc)
    # return now + datetime.timedelta(minutes=10 - now.minute % 10) - datetime.timedelta(
    #     seconds=now.second,
    #     microseconds=now.microsecond)
    return (
        now
        + datetime.timedelta(minutes=5 - now.minute % 5)
        - datetime.timedelta(seconds=now.second, microseconds=now.microsecond)
    )


def cli_entry():
    metadata = local_metadata()
    g = Validator(metadata)
    if exit_early():
        return
    asyncio.run(g.run())
    return


if __name__ == "__main__":
    cli_entry()
