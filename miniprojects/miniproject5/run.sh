#!/bin/sh
python Di_r.py _datasets/_dataset_21382/
python Di_r.py _datasets/_dataset_41981/
python Di_r.py _datasets/_dataset_57444/

python Dj_Di.py _datasets/_dataset_21382/ _datasets/_dataset_41981/
python Dj_Di.py _datasets/_dataset_21382/ _datasets/_dataset_57444/
python Dj_Di.py _datasets/_dataset_41981/ _datasets/_dataset_21382/
python Dj_Di.py _datasets/_dataset_41981/ _datasets/_dataset_57444/
python Dj_Di.py _datasets/_dataset_57444/ _datasets/_dataset_41981/
python Dj_Di.py _datasets/_dataset_57444/ _datasets/_dataset_21382/
