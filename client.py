#from src.pipe.Connectors import AWSConnector
#from src.pipe.DeePipe import DeePipe

import argparse
from DeePipe.components.DeePipe import DeePipe as dp

def consume_model():
    
    images = [
        'data/MNISTMini/2/35.jpg',
        'data/MNISTMini/4/6.jpg',
        'data/MNISTMini/7/17.jpg'
    ]

    pipe = dp()
    res = pipe.predict(project_name="MyProject", endpoint='MOPCTestEndpoint', images_path=images)
    print(res)


def mode_1(config_dict):
    ### OPTION 1 ###
    print("Option 1")
    pipe = dp()
    pipe.init(config_file=config_dict)


def mode_2():
    ### OPTION 2 ###
    print("Option 2")
    pipe = dp()
    pipe.init(name='ImageClassificationV2', task='classification')
    pipe.preproc_data(location='data/MNISTMini/', img_res=[28,28], greyscale=False, test_size=0.2, folds=2)
    pipe.train(max_epochs=1, batch_size=[64, 128], optimizer='Adam', learning_rate=[0.0001, 0.01], number_trials=3)
    pipe.eval()
    pipe.deploy()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Pipe NN.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    mode_1(args.config)
    #mode_2()

    #consume_model()
    
