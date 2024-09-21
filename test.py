import argparse
import time

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report

from dataloading import NewsDataModule
from main import NewsModel


def test_model(model, dm):
    trainer = pl.Trainer(accelerator="gpu",
                         devices=[5],
                         )
    val_loss = trainer.validate(model=model, datamodule=dm)[0]["val_loss"]
    test_loss = trainer.test(model=model, datamodule=dm)[0]["test_loss"]
    y_pred = np.array(model.test_preds).reshape(-1, 1)
    y_true = dm.test_dataset.targets.values.reshape(-1, 1)
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    start_time = time.time()
    pl.seed_everything(1234)
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_identifier", default='REDUCED_TRAINING_DATA')
    parser.add_argument("--article",
                        default='The game was tied 1-1 after the first period, but the Flyers outshot Buffalo 16-6 in the second and scored three times on goals by Fedotenko, Simon Gagne and John LeClair.',
                        help="Input article for prediction.")
    parser.add_argument("--only_predict", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="If provided, only make predictions without testing the model.")
    args = parser.parse_args()
    model_identifier = args.model_identifier
    model = NewsModel.load_from_checkpoint(f'saved/{model_identifier}.ckpt')
    dm = NewsDataModule("data/bbcsport", model.hparams.batch_size, model.hparams.max_token_len,
                        dataloader_pin_memory=model.hparams.single_gpu, dataloader_num_workers=model.hparams.num_workers)

    if not args.only_predict:
        test_model(model, dm)
    print(f"\nArticle: {args.article}")
    predicted_class = model.predict(args.article, dm.tokenizer, dm.class_labels)
    print(f"Predicted category: {predicted_class}")

    time_sec = time.time() - start_time
    time_min = time_sec / 60
    print("\nProcessing time of %s: %.2f seconds (%.2f minutes)."
    %("Test", time.time() - start_time, time_min))
