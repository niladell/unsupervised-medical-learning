NODE=1
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-1\n'
sleep 0.1

NODE=2
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-2\n'
sleep 0.1

NODE=3
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=BASIC --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-3\n'
sleep 0.1

NODE=4
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True
--tpu=node-4\n'
sleep 0.1

NODE=5
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5
--tpu=node-5\n'
sleep 0.1

NODE=5
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=BASIC --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5
--tpu=node-5\n'
sleep 0.1

NODE=6
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.1 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.1
--tpu=node-6\n'
sleep 0.1

NODE=7
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--use_wgan=True
--tpu=node-7\n'
sleep 0.1

NODE=8
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--noise_cov=POWER
--tpu=node-8\n'
sleep 0.1

NODE=9
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=3.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=3.0
--use_wgan=True
--tpu=node-9\n'
sleep 0.1

NODE=10
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--noise_cov=POWER
--tpu=node-10\n'
sleep 0.1

NODE=11
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_17 --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=2.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=2.0 \
--noise_cov=NORMAL --use_wgan=True
--tpu=node-11\n'
sleep 0.1

# CQ

NODE=12
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-12\n'
sleep 0.1

NODE=13
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-13\n'
sleep 0.1

NODE=14
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-14\n'
sleep 0.1

NODE=15
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--tpu=node-15\n'
sleep 0.1

NODE=15
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=1.0 \
--tpu=node-15\n'
sleep 0.1

NODE=16
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=1.0 \
--use_wgan=True \
--tpu=node-16\n'
sleep 0.1

NODE=17
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=1.0 \
--use_wgan=True \
--tpu=node-17\n'
sleep 0.1

NODE=18
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=0.1 --encoder=ATTACHED \
--reconstruction_loss=False --r_loss_lambda=0.0 \
--use_window_loss=True
--tpu=node-18\n'
sleep 0.1

NODE=19
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--use_window_loss=True --use_wgan=True --window_lambda=0.5
--tpu=node-19\n'
sleep 0.1

NODE=20
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=5.0 \
--use_window_loss=True --use_wgan=False --window_lambda=1.0
--tpu=node-20\n'
sleep 0.1

NODE=21
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--use_window_loss=True --use_wgan=False --window_lambda=0.2
--tpu=node-21\n'
sleep 0.1

NODE=22
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=64 --d_optimizer=ADAM --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=2.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=2.0 \
--use_window_loss=True --use_wgan=True --window_lambda=0.5
--tpu=node-22\n'
sleep 0.1

NODE=23
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.1 \
--use_window_loss=True --use_wgan=False --window_lambda=0.5
--tpu=node-23\n'
sleep 0.1

NODE=24
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=10 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=256 --d_optimizer=SGD --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--use_window_loss=True --use_wgan=False --window_lambda=0.5
--tpu=node-24\n'
sleep 0.1

NODE=25
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_17 --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=256 --d_optimizer=SGD --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=2.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=5.0 \
--use_window_loss=True --use_wgan=False --window_lambda=3.0
--tpu=node-25\n'
sleep 0.1
