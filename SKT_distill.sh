python SKT_distill.py A_train.json A_val.json A_test.json \
	--lr 1e-4 \
	-tc '/teacher/model/ckeckpoint' \
	-laf 0.5 \
	-lac 0.5 \
	--out /save/path/to/output