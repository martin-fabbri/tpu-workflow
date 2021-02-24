# tpu-worflows

## DVC

### Remote origin setup
```bash
dvc remote add origin https://dagshub.com/martin-fabbri/tpu-workflow.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USER"
dvc remote modify origin --local password "$SUPER_SECRET_PASSWORD"
```

### Define pipeline stages
```bash
dvc run -n split \
-d src/split.py \
-o data/interim/train_split.json \
-o data/interim/val_split.json \
python3 src/split.py --gcs-path gs://kds-357fde648f21ba86b09520d51e296ad06846fd421d364336db3d426d --batch-size 16 
```



