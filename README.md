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

### GCS
```
dvc run -n download_file \
-d gs://kaggle-data-tpu/test/test.txt \
-o test.txt \
gsutil cp gs://kaggle-data-tpu/test/test.txt test.txt
```

#### Remote aliases
```
['remote "gcs_test"']
    url = gs://kaggle-data-tpu/test
```
| gs://kaggle-data-tpu/test/test_2.txt

```bash
dvc run -n download_file_2 \
          -d remote://gcs_test/test_2.txt \
          -o test_2.txt \
          gsutil cp gs://kaggle-data-tpu/test/test_2.txt test_2.txt
```
```yaml
download_file_2:
    cmd: gsutil cp gs://kaggle-data-tpu/test/test_2.txt test_2.txt
    deps:
    - remote://gcs_test/test_2.txt
    outs:
    - test_2.txt
```

#### Test python script
```python
import logging
import os
from google.cloud import storage

client = storage.Client()

bucket = client.get_bucket("kaggle-data-tpu")
blob = bucket.get_blob("test/test_2.txt")
print(blob.download_as_string())
```

#### Test task stage
```bash
dvc run -n test_read_blob_gcs \
          -d remote://gcs_test/test_2.txt \
          python src/test.py
```

#### List objects in a bucket

```python
from google.cloud import storage

storage_client = storage.Client()
blobs = storage_client.list_blobs("gs://kaggle-data-tpu/test/")
for blob in blobs:
    print(blob.name)
```

```bash
dvc run -n test_list_objects_gcs \
          -d remote://gcs_test/ \
          python src/test_list_blobs.py
```



