import datetime
import os
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import RobertaModel

from dataloading import NewsDataModule, Configuration


class NewsModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.hparams.hidden_dim = int(self.hparams.hidden_dim)
        self.pre_classifier = torch.nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.dropout = nn.Dropout(p=self.hparams.drop)
        self.classifier = torch.nn.Linear(self.hparams.hidden_dim, 5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.test_preds = []
        self.val_preds = []

    def forward(self, token_ids, attention_mask):
        roberta_embeddings = self.roberta(input_ids=token_ids, attention_mask=attention_mask, )
        hidden_state = roberta_embeddings[0]
        pooler = hidden_state[:, 0]
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        pred = self.classifier(pooler)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                                    verbose=True),
            'monitor': 'val_loss',  # Metric to monitor
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        preds = self(batch["token_ids"], batch["token_mask"])
        targets = batch["targets"].long()
        loss = self.criterion(preds, targets)

        # Calculate accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == targets).float().sum()
        accuracy = correct / targets.size(0)

        # Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        preds = self(batch["token_ids"], batch["token_mask"])
        targets = batch["targets"].long()
        loss = self.criterion(preds, targets)

        # Calculate accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == targets).float().sum()
        accuracy = correct / targets.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_preds.extend([item[0] for item in preds.tolist()])
        return {"val_loss": loss, "val_acc": accuracy}

    def test_step(self, batch, batch_idx):
        preds = self(batch["token_ids"], batch["token_mask"])
        targets = batch["targets"].long()
        loss = self.criterion(preds, targets)
        # Calculate accuracy
        _, predicted = torch.max(preds, 1)
        correct = (predicted == targets).float().sum()
        accuracy = correct / targets.size(0)
        self.test_preds.extend(predicted.tolist())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_acc": accuracy}

    def predict(self, sentence, tokenizer, class_labels):
        inputs = tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.hparams.max_token_len,
            padding='max_length',
            truncation=True,
        )
        token_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
        token_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self(token_ids, token_mask)
            _, predicted = torch.max(preds, 1)
            predicted_class = class_labels[predicted.item()]

        return predicted_class

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--lr", default=1e-05, type=float)
        parser.add_argument("--drop", default=0.3, type=float)
        parser.add_argument('--max_token_len', default=256, type=int)
        parser.add_argument("--hidden_dim", default='768',
                            type=str)
        return parent_parser


start_time = time.time()
if __name__ == '__main__':
    pl.seed_everything(1234)
    warnings.filterwarnings(action='ignore')
    torch.set_float32_matmul_precision('medium')
    save_path = "saved"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--devices', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--single_gpu", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--configuration', default=Configuration.SYNTHETIC_TEST.name)
    parser = NewsModel.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.single_gpu:
        args.devices = 1
        args.batch_size = 64

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H-%M')
    model_identifier = f"{timestamp} config={args.configuration} lr={args.lr} bs={args.batch_size} epochs={args.epochs} drop={args.drop} num_workers={args.num_workers}"

    dm = NewsDataModule("data/bbcsport", args.batch_size, args.max_token_len, dataloader_pin_memory=args.single_gpu,
                        dataloader_num_workers=args.num_workers, configuration=Configuration[args.configuration])
    model = NewsModel(**args.__dict__)

    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor="val_loss",
                                          mode="min",
                                          dirpath=save_path,
                                          filename=f"model_{model_identifier}",
                                          enable_version_counter=False,
                                          )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step', )

    trainer = pl.Trainer(num_sanity_val_steps=0,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
                         accelerator="gpu",
                         devices=[5] if args.single_gpu else args.devices,
                         strategy='ddp_find_unused_parameters_true' if args.devices > 1 else 'auto',
                         )

    start_time_train = time.time()
    trainer.fit(model, datamodule=dm)

    time_sec = time.time() - start_time_train
    time_min = time_sec / 60
    print("\nProcessing time of %s: %.2f seconds (%.2f minutes).\n"
          % ("training", time.time() - start_time_train, time_min))

    val_loss = trainer.validate(model=model, datamodule=dm)[0]["val_loss"]
    test_loss = trainer.test(model=model, datamodule=dm)[0]["test_loss"]
    y_pred = np.array(model.test_preds).reshape(-1, 1)
    y_true = dm.test_dataset.targets.values.reshape(-1, 1)

    print()
    print(classification_report(y_true, y_pred))

    test_accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Precision: {precision} Recall: {recall} F1: {f1}")
    time_sec = time.time() - start_time
    time_min = time_sec / 60
    print("\nProcessing time of %s: %.2f seconds (%.2f minutes)."
          % ("Whole code", time.time() - start_time, time_min))

    model_config = args.configuration
    results_df = pd.DataFrame({
        'Acc (%)': [round(test_accuracy * 100, 2)],
        'F1': [round(f1 * 100, 2)],
        'Precision': [round(precision * 100, 2)],
        'Recall': [round(recall * 100, 2)],
        'Test Loss': [round(test_loss, 3)],
        'Val Loss': [round(val_loss, 2)]
    }, index=[model_config])
    results_filename = 'Results.csv'
    if not os.path.exists(results_filename):
        results_df.to_csv(results_filename)
    else:
        existing_df = pd.read_csv(results_filename, index_col=0)
        updated_df = pd.concat([existing_df, results_df])
        updated_df.to_csv(results_filename, header=True)
