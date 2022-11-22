import tensorflow as tf
import config as cfg
from tools import callback, optimizer,  loss_fn, pesq, stoi
from dataloader import steps_val, steps_train, dataset_val, dataset_train
from model import NUNET_LSTM
import os

if __name__ == "__main__":
    if os.path.isdir(cfg.chpt_dir) is False:
        print("[Error] There is no directory '%s'." % cfg.chpt_dir)
        os.system('mkdir ' + str(cfg.chpt_dir))

    m = NUNET_LSTM()
    model = m.train_model()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[pesq, stoi], run_eagerly=False)

    # model.load_weights('Input path of weights (saved model, h5..)')

    model.fit(dataset_train,
              batch_size=None,
              steps_per_epoch=steps_train,
              epochs=cfg.EPOCH,
              # initial_epoch=None,
              verbose=1,
              validation_data=dataset_val,
              validation_steps=steps_val,
              callbacks=callback,
              max_queue_size=50,
              workers=4,
              use_multiprocessing=True
              )

    model_path = "./saved_model/" + cfg.file_name

    model.save_weights(model_path)
    tf.keras.backend.clear_session()
