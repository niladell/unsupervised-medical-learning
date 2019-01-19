NODE=1
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_finale --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-1\n'
sleep 0.1

NODE=2
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_finale --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-2\n'
sleep 0.1

NODE=3
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_finale --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=BASIC --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-3\n'
sleep 0.1

NODE=4
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_finale --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True \
--tpu=node-4\n'
sleep 0.1

NODE=5
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_finale --data_dir=gs://iowa_bucket/celebA/data/ --dataset=celebA \
--iterations_per_loop=200 --batch_size=1024  --train_steps=100000 --train_steps_per_eval=2000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--tpu=node-5\n'
sleep 0.1




NODE=12
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-12\n'
sleep 0.1

NODE=13
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=64 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-13\n'
sleep 0.1

NODE=14
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--tpu=node-14\n'
sleep 0.1

NODE=15
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=BASIC --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=0.5 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=0.5 \
--tpu=node-15\n'
sleep 0.1

NODE=16
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=1.0 \
--use_wgan=True \
--tpu=node-16\n'
sleep 0.1

NODE=17
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=SGD --learning_rate=0.0002 \
--use_encoder=True --e_loss_lambda=1.0 --encoder=ATTACHED \
--reconstruction_loss=True --r_loss_lambda=1.0 \
--use_wgan=True \
--tpu=node-17\n'
sleep 0.1

NODE=18
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/CQ500/bulk_finale --data_dir=gs://iowa_bucket/CQ500/data/train_256.tfrecords --dataset=CQ500_256 \
--iterations_per_loop=25 --batch_size=32  --train_steps=100000 --train_steps_per_eval=1000 \
--model=RESNET --noise_dim=100 --d_optimizer=ADAM --learning_rate=0.0001 \
--use_encoder=True --e_loss_lambda=0.1 --encoder=ATTACHED \
--reconstruction_loss=False --r_loss_lambda=0.0 \
--use_window_loss=True \
--tpu=node-18\n'
sleep 0.1
