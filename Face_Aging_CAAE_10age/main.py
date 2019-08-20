import tensorflow as tf
from FaceAging import FaceAging
from os import environ
import argparse
import logging

# environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--dataset', type=str, default='TrainingSet_CACD2000', help='training dataset name that stored in ./data')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='test', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=True, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=True, help='whether train from the init model if cannot find an existing model')
FLAGS = parser.parse_args()


def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS)

    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    #create logfile by linda
    logger = createlog()
    logger.info('main function start...')

    with tf.Session(config=config) as session:
        model = FaceAging(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            dataset_name=FLAGS.dataset,  # name of the dataset in the folder ./data
            log = logger
        )
        if FLAGS.is_train:
            logger.info('\n\tTraining Mode')
            print ('\n\tTraining Mode')
            if not FLAGS.use_trained_model:
                print ('\n\tPre-train the network')
                model.train(
                    num_epochs=10,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    use_init_model=FLAGS.use_init_model,
                    weigts=(0, 0, 0)
                )
                print ('\n\tPre-train is done! The training will start.')
            model.train(
                num_epochs=FLAGS.epoch,  # number of epochs 50
                use_trained_model=FLAGS.use_trained_model,
                use_init_model=FLAGS.use_init_model
            )
        else:
            print ('\n\tTesting Mode')
            print ('\n\tTesting Mode'+FLAGS.testdir + '/*jpg')
            model.custom_test(
                testing_samples_dir=FLAGS.testdir + '/*jpg'
            )

def createlog():
    # log fine
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # a file handler
    handler = logging.FileHandler('output.log')
    handler.setLevel(logging.INFO)

    # a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    tf.compat.v1.app.run()