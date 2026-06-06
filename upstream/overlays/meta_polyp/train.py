import os
import tensorflow as tf
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, total_loss
from tensorflow.keras.utils import get_custom_objects
from callbacks.callbacks import get_callbacks, cosine_annealing_with_warmup
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
from model import build_model
import tensorflow_addons as tfa
from optimizers.lion_opt import Lion

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = 256
BATCH_SIZE = 8
SEED = 42
save_path = "best_model.h5"
epochs = 200
save_weights_only = True
max_lr = 1e-4
min_lr = 1e-6

model = build_model(img_size)

starter_learning_rate = 1e-4
end_learning_rate = 1e-6
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.2)

opts = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=learning_rate_fn)

get_custom_objects().update({"dice": dice_loss})
model.compile(optimizer=opts,
              loss='dice',
              metrics=[dice_coeff, bce_dice_loss, IoU, zero_IoU])

train_route = '../../data/kvasir/train'
val_route = '../../data/kvasir/val'
test_route = '../../data/kvasir/test'

X_train = sorted([train_route + '/images/' + x for x in os.listdir(train_route + '/images')])
Y_train = sorted([train_route + '/masks/' + y for y in os.listdir(train_route + '/masks')])
X_valid = sorted([val_route + '/images/' + x for x in os.listdir(val_route + '/images')])
Y_valid = sorted([val_route + '/masks/' + y for y in os.listdir(val_route + '/masks')])
X_test = sorted([test_route + '/images/' + x for x in os.listdir(test_route + '/images')])
Y_test = sorted([test_route + '/masks/' + y for y in os.listdir(test_route + '/masks')])

print("N Train:", len(X_train))
print("N Valid:", len(X_valid))
print("N Test:", len(X_test))

train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder,
                              augmentAdv=False, augment=False, augmentAdvSeg=True)

valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder,
                              augmentAdv=False, augment=False, repeat=False, shuffle=False,
                              augmentAdvSeg=False)

test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder,
                             augmentAdv=False, augment=False, repeat=False, shuffle=False,
                             augmentAdvSeg=False)

callbacks = get_callbacks(monitor='val_loss', mode='min', save_path=save_path, _max_lr=max_lr,
                          _min_lr=min_lr, _cos_anne_ep=1000, save_weights_only=save_weights_only)

steps_per_epoch = len(X_train) // BATCH_SIZE

print("START TRAINING:")
his = model.fit(train_dataset,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_data=valid_dataset)

model.load_weights(save_path)
model.evaluate(test_dataset)
model.save("final_model.h5")
