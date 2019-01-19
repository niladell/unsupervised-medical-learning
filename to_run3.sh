NODE=1
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-1 --dataset=celebA --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=2000\n'
sleep 0.1

NODE=2
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-2 --dataset=celebA --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=2000 --use_window_loss=False\n'
sleep 0.1

NODE=3
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=1 --batch_size=1024 --tpu=node-3 --dataset=celebA --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=2000 --use_window_loss=False --reconstruction_loss=True\n'
sleep 0.1

NODE=4
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=5 --batch_size=1024 --tpu=node-4 --use_wgan=True --dataset=celebA --model=RESNET --noise_dim=1200 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --wgan_n=1\n'
sleep 0.1

NODE=5
#screen -r echo "node-${NODE} on!" ||
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-5 --use_wgan=True --dataset=celebA --model=RESNET --noise_dim=1200 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --wgan_n=1 --reconstruction_loss=True\n'
sleep 0.1

NODE=6
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-6 --use_wgan=True --dataset=celebA --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --wgan_n=5 --noise_cov=POWER --model=RESNET\n'
sleep 0.1

NODE=7
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-7 --use_wgan=True --dataset=celebA --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --wgan_lambda=5 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --wgan_n=5 --noise_cov=POWER --model=RESNET --reconstruction_loss=True\n'
sleep 0.1

NODE=8
#screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=10 --batch_size=16 --tpu=node-8 --dataset=celebA --model=BASIC --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=1000\n'
sleep 0.1

NODE=9
# screen -r echo "node-${NODE} on!" || screen -dmS node-$NODE
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-9 --dataset=celebA --model=BASIC --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=2000 --reconstruction_loss=True\n'
sleep 0.1

NODE=18
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-18 --dataset=celebA --model=RESNET --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

NODE=19
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-19 --dataset=celebA --model=RESNET --noise_dim=8 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --window_lambda=10.0 --reconstruction_loss=False --noise_cov=POWER\n'
sleep 0.1

NODE=20
screen -dmS node-$NODE
NODE=20
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=False --iterations_per_loop=200 --batch_size=1024 --tpu=node-20 --dataset=celebA --model=RESNET --noise_dim=8 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.5 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --reconstruction_loss=True\n'
sleep 0.1

NODE=21
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-21 --dataset=celebA --model=RESNET --noise_dim=8 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --window_lambda=10.0 --reconstruction_loss=True\n'
sleep 0.1

NODE=22
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-22 --dataset=celebA --model=RESNET --noise_dim=100 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --window_lambda=10.0 --reconstruction_loss=True\n'
sleep 0.1

NODE=25
screen -dmS node-$NODE
NODE=25
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-25 --dataset=celebA --model=RESNET --noise_dim=8 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --window_lambda=2.0 --reconstruction_loss=True\n'
sleep 0.1

### OHAMA

NODE=10
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-10 --dataset=celebA --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=1\n'
sleep 0.1

NODE=11
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-11 --dataset=celebA --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=1 --use_window_loss=False\n'
sleep 0.1

NODE=12
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=20 --batch_size=16 --tpu=node-12 --dataset=celebA --model=RESNET --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=1 --use_window_loss=False --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

NODE=13
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-13 --dataset=celebA --model=RESNET --noise_dim=256 --train_steps=500000 --d_optimizer=SGD --learning_rate=0.0002 --e_loss_lambda=1.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=1 --use_window_loss=False --window_lambda=1.0 --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

NODE=14
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-14 --dataset=celebA --model=BASIC --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=1 --use_window_loss=False --window_lambda=5.0 --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

NODE=15
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-15 --dataset=celebA --model=BASIC --noise_dim=128 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=2.0 --train_steps_per_eval=2000 --use_wgan=True --wgan_n=5 --wgan_lambda=10 --use_window_loss=False --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

NODE=16
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-16 --dataset=celebA --model=RESNET --noise_dim=64 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=1.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --reconstruction_loss=True\n'
sleep 0.1

NODE=17
screen -dmS node-$NODE
screen -r node-$NODE -p 0 -X stuff 'python src/main.py --model_dir=gs://iowa_bucket/celebA/bulk_16_02_256 --data_dir=gs://iowa_bucket/celebA/data/ --use_encoder=True --iterations_per_loop=200 --batch_size=1024 --tpu=node-17 --dataset=celebA --model=RESNET --noise_dim=64 --train_steps=500000 --d_optimizer=ADAM --learning_rate=0.0001 --e_loss_lambda=0.0 --train_steps_per_eval=2000 --use_wgan=False --use_window_loss=False --reconstruction_loss=True --noise_cov=POWER\n'
sleep 0.1

echo "done"
