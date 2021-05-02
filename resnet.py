import tensorflow as tf
import numpy as np
import pprint
import webdataset as wds
from webdataset import multi
import typer

pp = pprint.PrettyPrinter(indent=4).pprint

SHARDS = "imagenet-train-{000000..001281}.tar"

app = typer.Typer()


def preproc(sample):
    """Perform preprocessing and data augmentation."""

    # just create mock data for testing
    sample["jpg"] = np.zeros((224, 224, 3))
    sample["hot"] = np.zeros(1000)
    sample["hot"][sample["cls"]] = 1
    return sample


class ImagenetData:
    """This class is a convenient placeholder for the dataset-related information.
    You could also just define these iterator etc. as global functions."""

    def __init__(self, prefix="/shards/", shards=SHARDS):
        self.length = 1281000
        self.urls = prefix + shards
        self.dataset = wds.WebDataset(self.urls).decode("rgb").map(preproc).to_tuple("jpg", "hot")
        self.loader = multi.MultiLoader(self.dataset, workers=4)

    def __iter__(self):
        for img, hot in self.loader:
            yield img.astype("float32"), np.array(hot).astype("float32")

    def __len__(self):
        return self.length

    def output_shapes(self):
        return ((224, 224, 3), (1000,))

    def output_types(self):
        return (tf.float32, tf.float32)


@app.command()
def train(
    prefix: str = "/shards/", shards: str = SHARDS, batchsize: int = 64, epochs: int = 25, lr: float = 0.001,
):

    # get the dataset descriptor

    df = ImagenetData(prefix=prefix, shards=shards)

    # initialize the model

    model = tf.keras.applications.resnet.ResNet50(
        input_shape=df.output_shapes()[0], include_top=True, weights=None
    )

    # compile the model

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # create the input pipeline

    tdf = tf.data.Dataset.from_generator(
        generator=df.__iter__, output_types=df.output_types(), output_shapes=df.output_shapes()
    )
    tdf = tdf.batch(batchsize)
    tdf = tdf.prefetch(tf.data.experimental.AUTOTUNE)

    # perform training

    model.fit(tdf, epochs=epochs, steps_per_epoch=len(df) // batchsize)


if __name__ == "__main__":
    app()
