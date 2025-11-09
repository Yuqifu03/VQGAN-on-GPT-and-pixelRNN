VQGAN-on-GPT-and-pixelRNN
Project for CSC2503
VQ-GAN Training
Train the VQ-GAN model:
bashpython vqgan_transformer_fashionmnist.py --stage vqgan --epochs 10 --out_dir runs
This saves checkpoints under runs/ and logs sample reconstructions.
Encode the Dataset to Tokens
Encode the dataset using the trained VQ-GAN:
bashpython vqgan_transformer_fashionmnist.py --stage encode \
  --ckpt_vqgan runs/vqgan_final.pt --out_dir runs
Train the Transformer (GPT) on Tokens
Train the GPT model on the encoded tokens:
bashpython vqgan_transformer_fashionmnist.py --stage gpt \
  --train_codes runs/train_codes.pt --epochs 10 --out_dir runs
Generate Samples
Generate new samples using both models:
bashpython vqgan_transformer_fashionmnist.py --stage sample \
  --ckpt_vqgan runs/vqgan_final.pt \
  --ckpt_gpt runs/gpt_final.pt \
  --train_codes runs/train_codes.pt \
  --num_samples 16 --temperature 1.0 --top_k 64 --out_dir runs
Output image grid: runs/samples.png
