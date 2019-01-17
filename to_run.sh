OUTPUT_DIR=gs://iowa_bucket/CQ500/bulk_16_01

NODE=1
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-1 --dataset=cq500 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500\n'

NODE=2
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-2 --dataset=cq500 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500 --use_window_loss=False\n'

NODE=3
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-3 --dataset=cq500 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500 --use_window_loss=True --reconstruction_loss=True\n'

NODE=4
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-4 --use_wgan=True --dataset=cq500 --model=RESNET --noise_dim=1200 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=500 --wgan_n=1\n'

NODE=5
#screen -r echo "node-${NODE} on!" ||
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-5 --use_wgan=True --dataset=cq500 --model=RESNET --noise_dim=1200 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=500 --wgan_n=1 --reconstruction_loss=True\n'

NODE=6
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-6 --use_wgan=True --dataset=cq500 --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=500 --wgan_n=5 --noise_cov=POWER --model=RESNET\n'

NODE=7
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-7 --use_wgan=True --dataset=cq500 --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=500 --wgan_n=5 --noise_cov=POWER --model=RESNET --reconstruction_loss=True\n'

NODE=8
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-8 --dataset=cq500 --model=BASIC --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500\n'

NODE=9
# screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-9 --dataset=cq500 --model=BASIC --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500 --reconstruction_loss=True\n'

NODE=10
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-10 --dataset=cq500 --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1\n'

NODE=11
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-11 --dataset=cq500 --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=False\n'

NODE=12
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-12 --dataset=cq500 --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=True --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'

NODE=13
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-13 --dataset=cq500 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=True --window_lambda=1.0 --reconstruction_loss=True --noise_cov=POWER\n'

NODE=14
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01 --data_dir=gs://iowa_bucket/CQ500/data --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-14 --dataset=cq500 --model=BASIC --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=True --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'

####################3
#### 256 x 256

NODE=15
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01_256 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-15 --dataset=CQ500_256 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500\n'
NODE=16
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01_256 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-16 --dataset=CQ500_256 --model=RESNET --noise_dim=68 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_window_loss=True --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'

NODE=17
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01_256 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-17 --dataset=CQ500_256 --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=True --window_lambda=1.0 --reconstruction_loss=True --noise_cov=POWER\n'

NODE=18
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_16_01_256 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --use_encoder=True --iterations_per_loop=35 --batch_size=32 --tpu=node-18 --dataset=CQ500_256 --model=BASIC --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=500 --use_wgan=True --wgan_n=1 --use_window_loss=True --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'

