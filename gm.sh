python src/train_gm.py src/config
python src/test_gm.py src/config Etest test.probs &
python src/test_gm.py src/config Edev dev.probs &
wait 
python src/matrix2measures-headtail-ents.py src/config > measures-ents.txt &
python src/matrix2measures-headtail-types.py src/config > measures-types.txt &
wait

