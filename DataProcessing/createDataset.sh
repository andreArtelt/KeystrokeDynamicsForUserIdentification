#!/usr/bin/env sh
python convert_dbs2dfs.py "../Data/"
python create_dataset.py "../Data/" "../Data/data_10.npz" "time" 10
python create_dataset.py "../Data/" "../Data/data_20.npz" "time" 20
python create_dataset.py "../Data/" "../Data/data_30.npz" "time" 30
python create_dataset.py "../Data/" "../Data/data_40.npz" "time" 40
python create_dataset.py "../Data/" "../Data/data_50.npz" "time" 50
python create_dataset.py "../Data/" "../Data/data_60.npz" "time" 60
python create_dataset.py "../Data/" "../Data/data_70.npz" "time" 70
python defence_mechanisms.py "../Data/" "../Data/data_defence/"
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_10.npz" "time" 10
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_20.npz" "time" 20
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_30.npz" "time" 30
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_40.npz" "time" 40
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_50.npz" "time" 50
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_60.npz" "time" 60
python create_dataset.py "../Data/data_defence/" "../Data/data_defence_70.npz" "time" 70
python convert_dbs2dfs.py "../Data/" "time2"
rm -r "../Data/data_defence/"
python create_dataset.py "../Data/" "../Data/data_defence_defence_10.npz" "time2" 10
python create_dataset.py "../Data/" "../Data/data_defence_defence_20.npz" "time2" 20
python create_dataset.py "../Data/" "../Data/data_defence_defence_30.npz" "time2" 30
python create_dataset.py "../Data/" "../Data/data_defence_defence_40.npz" "time2" 40
python create_dataset.py "../Data/" "../Data/data_defence_defence_50.npz" "time2" 50
python create_dataset.py "../Data/" "../Data/data_defence_defence_60.npz" "time2" 60
python create_dataset.py "../Data/" "../Data/data_defence_defence_70.npz" "time2" 70