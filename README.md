# VQGAN-on-GPT-and-pixelRNN
project for CSC2503

VQ-GAN training

python vqgan_transformer_fashionmnist.py --stage vqgan --epochs 10 --out_dir runs


This saves checkpoints under runs/ and logs sample reconstructions.

Encode the dataset to tokens

python vqgan_transformer_fashionmnist.py --stage encode \
  --ckpt_vqgan runs/vqgan_final.pt --out_dir runs


Train the Transformer (GPT) on tokens

python vqgan_transformer_fashionmnist.py --stage gpt \
  --train_codes runs/train_codes.pt --epochs 10 --out_dir runs


Generate samples

python vqgan_transformer_fashionmnist.py --stage sample \
  --ckpt_vqgan runs/vqgan_final.pt \
  --ckpt_gpt runs/gpt_final.pt \
  --train_codes runs/train_codes.pt \
  --num_samples 16 --temperature 1.0 --top_k 64 --out_dir runs


Output image grid: runs/samples.png.
