
This code generates tfrecords for the inception model
'''bash
python build_image_data.py --train_directory=../dataset/train/ --validation_directory=../dataset/hand-labeled/ --output_directory=dataset/ --labels_file=labels.txt
'''

Model download link: https://www.deepdetect.com/models/tf/inception_v4.pb
checkpoint file for the pretrained network: http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz


run command: python retrain.py --model_dir=inception-v4-model/ --image_dir=../dataset/train/ --architecture=inception_v4 --output_graph=output_graph/output_graph.pb --intermediate_output_graphs_dir=intermediate_output_graphs_dir/ --output_labels=output_labels/output_labels.txt --summaries_dir=summaries_dir/ --bottleneck_dir=bottleneck_dir/ --train_batch_size=32 --learning_rate=0.001 --how_many_trainin_steps=50000
