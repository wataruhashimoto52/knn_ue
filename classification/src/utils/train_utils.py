import logging
from typing import Union

import numpy as np
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def apply_dropout(m: nn.Module):
    if type(m) == nn.Dropout:
        m.train()


class EarlyStoppingForTransformers:
    def __init__(
        self,
        path: str,
        patience: int = 5,
        verbose: bool = False,
        optimize_type: str = "maximize",
    ):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience  # 設定ストップカウンタ
        self.verbose = verbose  # 表示の有無
        self.counter = 0  # 現在のカウンタ値
        self.best_score = None  # ベストスコア
        self.early_stop = False  # ストップフラグ
        self.val_loss_min = np.Inf  # 前回のベストスコア記憶用
        self.optimize_type = optimize_type
        self.path = path  # ベストモデル格納path

    def __call__(
        self,
        val_loss: float,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        save_flag: bool = True,
    ) -> bool:
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = val_loss

        if self.best_score is None:  # 1Epoch目の処理
            self.best_score = score  # 1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(
                val_loss, model, tokenizer, save_flag
            )  # 記録後にモデルを保存してスコア表示する
            return True

        elif self.optimize_type == "maximize":
            if score < self.best_score:  # ベストスコアを更新できなかった場合
                self.counter += 1  # ストップカウンタを+1
                if self.verbose:  # 表示を有効にした場合は経過を表示
                    logger.info(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )  # 現在のカウンタを表示する
                if self.counter >= self.patience:  # 設定カウントを上回ったらストップフラグをTrueに変更
                    self.early_stop = True

                return False

            else:  # ベストスコアを更新した場合
                self.best_score = score  # ベストスコアを上書き
                # self.checkpoint(val_loss, model, tokenizer, save_flag)  # モデルを保存してスコア表示
                self.counter = 0  # ストップカウンタリセット
                return True

        elif self.optimize_type == "minimize":
            if score > self.best_score:  # ベストスコアを更新できなかった場合
                self.counter += 1  # ストップカウンタを+1
                if self.verbose:  # 表示を有効にした場合は経過を表示
                    logger.info(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )  # 現在のカウンタを表示する
                if self.counter >= self.patience:  # 設定カウントを上回ったらストップフラグをTrueに変更
                    self.early_stop = True
                return False

            else:  # ベストスコアを更新した場合
                self.best_score = score  # ベストスコアを上書き
                # self.checkpoint(val_loss, model, tokenizer, save_flag)  # モデルを保存してスコア表示
                self.counter = 0  # ストップカウンタリセット
                return True
        else:
            raise NotImplementedError()

    def checkpoint(
        self,
        val_loss: float,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        save_flag: bool = True,
    ):
        """ベストスコア更新時に実行されるチェックポイント関数"""
        if self.verbose:  # 表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            logger.info(
                f"Validation score updated. ({str(self.val_loss_min)} --> {str(val_loss)})."
            )
        # Save Model
        if save_flag:
            logger.info("Saving model...")
            model.save_pretrained(self.path)
            tokenizer.save_pretrained(self.path)

        self.val_loss_min = val_loss  # その時のlossを記録する
