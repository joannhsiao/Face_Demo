# Face_Demo

## versions
- Face_efm_v2.py: loss only triplet loss, no accuarcy, only save params
- Face_efm_v3.py: total loss = id loss + alpha * triplet loss, have accuracy, export model(architecture) and params

- lightcnn.py: only output feacture vector through weight matrix, no output
- lightcnn_v2.py: output weight matrix and id output (ans)

- EFM_triplet_symbol.py: using symbol rather than gluon

---
## testing
### gluon
- only triplet loss (MNIST): well
- loss = id loss + 0.1 * triplet loss: not so good
- loss = id loss + 0.5 * triplet loss: 
- loss = id loss + triplet loss: 

---
## Run
### training
- asian celeb: 
	```
	./train_asianceleb.sh
	```
- celeb1M: 
	```
	./train_efm.sh
	```

### figures
- cosine similarity: `python draw_cos_dis_real.py epoch`
