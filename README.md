**This project is a prototype for experimental purposes only and production grade code is not released here.**

# Deep LSTM siamese network for text similarity

It is a tensorflow based implementation of deep siamese LSTM network to capture phrase/sentence similarity using character embeddings.

This code provides architecture for learning two kinds of tasks:

- Phrase similarity using char level embeddings [1]
![siamese lstm phrase similarity](https://cloud.githubusercontent.com/assets/9861437/20479454/405a1aea-b004-11e6-8a27-7bb05cf0a002.png)

- Sentence similarity using word level embeddings [2]
![siamese lstm sentence similarity](https://cloud.githubusercontent.com/assets/9861437/20479493/6ea8ad12-b004-11e6-89e4-53d4d354d32e.png)

For both the tasks mentioned above it uses a multilayer siamese LSTM network and euclidian distance based contrastive loss to learn input pair similairty.

# Capabilities
Given adequate training pairs, this model can learn Semantic as well as structural similarity. For eg:

**Phrases :**
- International Business Machines = I.B.M
- Synergy Telecom = SynTel
- Beam inc = Beam Incorporate
- Sir J J Smith = Johnson Smith
- Alex, Julia = J Alex
- James B. D. Joshi	= James Joshi
- James Beaty, Jr. = Beaty

For phrases, the model learns **character based embeddings** to identify structural/syntactic similarities.

**Sentences :**
- He is smart = He is a wise man.
- Someone is travelling countryside = He is travelling to a village.
- She is cooking a dessert = Pudding is being cooked.
- Microsoft to acquire Linkedin ≠ Linkedin to acquire microsoft

(More examples Ref: semEval dataset)

For Sentences, the model uses **pre-trained word embeddings** to identify semantic similarities.

Categories of pairs, it can learn as similar:
- Annotations
- Abbreviations
- Extra words
- Similar semantics
- Typos
- Compositions
- Summaries

# Training Data
- **Phrases:** 
	- A sample set of learning person name paraphrases have been attached to this repository. To generate full person name disambiguation data follow the steps mentioned at:

	> https://github.com/dhwajraj/dataset-person-name-disambiguation

    "person_match.train" : https://drive.google.com/open?id=1HnMv7ulfh8yuq9yIrt_IComGEpDrNyo-
- **Sentences:** 
	- A sample set of learning sentence semantic similarity can be downloaded from:

	"train_snli.txt" : https://drive.google.com/open?id=1itu7IreU_SyUSdmTWydniGxW-JEGTjrv

	This data is generated using SNLI project : 
	> https://nlp.stanford.edu/projects/snli/

	 - word embeddings: any set of pre-trained word embeddings can be utilized in this project. For our testing we had used fastText 	simple english embeddings from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

	alternate download location for "wiki.simple.vec" is : https://drive.google.com/open?id=1u79f3d2PkmePzyKgubkbxOjeaZCJgCrt

# Environment
- numpy 1.11.0
- tensorflow 1.2.1
- gensim 1.0.1
- nltk 3.2.2

# How to run
### Training
```
$ python train.py [options/defaults]

options:
  -h, --help            show this help message and exit
  --is_char_based IS_CHAR_BASED
  			is character based syntactic similarity to be used for phrases.
			if false then word embedding based semantic similarity is used.
			(default: True)
  --word2vec_model WORD2VEC_MODEL
    			this flag will be used only if IS_CHAR_BASED is False
  			word2vec pre-trained embeddings file (default: wiki.simple.vec)
  --word2vec_format WORD2VEC_FORMAT
  			this flag will be used only if IS_CHAR_BASED is False
  			word2vec pre-trained embeddings file format (bin/text/textgz)(default: text)
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 100)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --max_document_words MAX_DOCUMENT_WORDS
                        Max length (left to right max words to consider) in
                        every doc, else pad 0 (default: 100)
  --training_files TRAINING_FILES
                        Comma-separated list of training files (each file is
                        tab separated format) (default: None)
  --hidden_units HIDDEN_UNITS
                        Number of hidden units(default:50)
  --batch_size BATCH_SIZE
                        Batch Size (default: 128)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 200)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 2000)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 2000)
  --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
  --nolog_device_placement

```
### Evaluation
```
$ python eval.py --model graph#.pb
```

# Performance
**Phrases:**
- Training time: (8 core cpu) = 1 complete epoch : 6min 48secs (training requires atleast 30 epochs)
	- Contrastive Loss : 0.0248
- Evaluation performance : similarity measure for 100,000 pairs (8core cpu) = 1min 40secs
	- Accuracy 91%
	
**Sentences:**
- Training time: (8 core cpu) = 1 complete epoch : 8min 10secs (training requires atleast 50 epochs)
	- Contrastive Loss : 0.0477
- Evaluation performance : similarity measure for 100,000 pairs (8core cpu) = 2min 10secs
	- Accuracy 81%

# References
1. [Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W16-16#page=162)
2. [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
