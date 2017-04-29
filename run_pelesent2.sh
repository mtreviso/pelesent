#!/bin/bash
# rodar: sudo ./run.sh

sudo python3 -m pelesent --gpu --pos-file data/corpora/CLEAN_buscape2.pos --neg-file data/corpora/CLEAN_buscape2.neg --emb-file data/embs/w2v-twitter-skip-300.model --emb-type word2vec


invokeps(){
	local pf=$1
	local nf=$2
	local ef=$3
	local et=$4
	time sudo python3 -m pelesent --gpu --pos-file $pf --neg-file $nf --emb-file $ef --emb-type $et
}


# 
invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec
invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec

invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec
invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec

invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec
invokeps data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg data/embs/w2v-twitter-skip-300.model word2vec
