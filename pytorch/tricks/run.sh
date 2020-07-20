CUDA_VISIBLE_DEVICES=1 python trainer.py  --use-amp --use-warmup --use-labelsmooth --omega=0.7
CUDA_VISIBLE_DEVICES=1 python extractor.py
python retrieval_in_category.py