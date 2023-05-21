import paddle
from numpy import round
from tqdm import tqdm


def get_paddlepaddle_cifar10_data(batch_size=128) -> dict:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
       CIFAR10 dataloader
    """
    # Load and transform the CIFAR10 data
    # Each batch will yield 128 images
    buf_size = 50000
    # Reader for training
    train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=buf_size),
        batch_size=batch_size,
    )

    # Reader for testing. A separated data set for testing.
    test_reader = paddle.batch(paddle.dataset.cifar.test10(), batch_size=batch_size)
    return {"train_reader": train_reader, "test_reader": test_reader, "buf_size": buf_size}


def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return paddle.fluid.nets.img_conv_group(
            input=ipt,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act="relu",
            conv_with_batchnorm=False,
            conv_batchnorm_drop_rate=dropouts,
            pool_type="max",
            pool_size=2,
            pool_stride=2,
        )

    conv1 = conv_block(input, 16, 2, [0.3, 0])
    conv2 = conv_block(conv1, 32, 2, [0.4, 0])
    fc1 = paddle.fluid.layers.fc(input=conv2, size=64, act="relu")
    predict = paddle.fluid.layers.fc(input=fc1, size=10, act="softmax")
    return predict


def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = paddle.fluid.layers.data(name="pixel", shape=data_shape, dtype="float32")

    predict = vgg_bn_drop(images)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


def train_program():
    predict = inference_program()

    label = paddle.fluid.layers.data(name="label", shape=[1], dtype="int64")
    cost = paddle.fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = paddle.fluid.layers.mean(cost)
    accuracy = paddle.fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy, predict]


def optimizer_program():
    return paddle.fluid.optimizer.Adam(learning_rate=0.001)


def train_test(program, reader, feed_order, place, avg_cost, acc):
    count = 0
    feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]
    feeder_test = paddle.fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = paddle.fluid.Executor(place)
    accumulated = len([avg_cost, acc]) * [0]
    for test_data in reader():
        avg_cost_np = test_exe.run(
            program=program,
            feed=feeder_test.feed(test_data),
            fetch_list=[avg_cost, acc],
        )
        accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]
        count += 1
    return [x / count for x in accumulated]


def run_paddlepaddle_cifar10_training(
    dataloader: dict,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with paddlepaddle frameworks.

    Returns:
        validation accuracy
    """
    use_cuda = False
    place = paddle.fluid.CUDAPlace(0) if use_cuda else paddle.fluid.CPUPlace()
    paddle.enable_static()

    feed_order = ["pixel", "label"]

    main_program = paddle.fluid.default_main_program()
    star_program = paddle.fluid.default_startup_program()

    avg_cost, acc, predict = train_program()

    # Test program
    test_program = main_program.clone(for_test=True)

    optimizer = optimizer_program()
    optimizer.minimize(avg_cost)

    exe = paddle.fluid.Executor(place)
    params_dirname = "image_classification_resnet.inference.model"

    feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
    feeder = paddle.fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)

    train_reader, test_reader, buf_size = (
        dataloader["train_reader"],
        dataloader["test_reader"],
        dataloader["buf_size"],
    )
    for pass_id in range(epochs):
        progress_bar = tqdm(
            train_reader(),
            position=0,
            leave=True,
            total=round(buf_size / batch_size),
        )
        for data_train in train_reader():
            avg_loss_value = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[avg_cost, acc],
            )

            progress_bar.set_description(f"Epoch {pass_id+1}/{epochs}")
            progress_bar.set_postfix(train_loss=round(avg_loss_value[0], 3))
            progress_bar.update()

        _, accuracy_test = train_test(test_program, test_reader, feed_order, place, avg_cost, acc)

        # save parameters
        if params_dirname is not None:
            paddle.fluid.io.save_inference_model(params_dirname, ["pixel"], [predict], exe)
    return accuracy_test
