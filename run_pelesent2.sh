#!/bin/bash
# rodar: sudo ./run.sh

invokeps(){
	local pf=$1
	local nf=$2
	local ef=$3
	local et=$4
	time sudo python3 -m pelesent --gpu --pos-file $pf --neg-file $nf --emb-file $ef --emb-type $et
}

invokeds(){
	local dpos=$1
	local dneg=$2
	
	# 50dim 
	# ----------------------------------
	# sg
	# invokeps $dpos $dneg data/embs/w2v-twitter-skip-50.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-skip-50.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-skip-50.model word2vec

	# # # cbow
	# invokeps $dpos $dneg data/embs/w2v-twitter-cbow-50.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-cbow-50.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-cbow-50.model word2vec

	# # # glove
	# invokeps $dpos $dneg data/embs/glove-twitter-50.model word2vec



	# # 100dim 
	# # ----------------------------------
	# # sg
	# invokeps $dpos $dneg data/embs/w2v-twitter-skip-100.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-skip-100.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-skip-100.model word2vec

	# # cbow
	# invokeps $dpos $dneg data/embs/w2v-twitter-cbow-100.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-cbow-100.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-cbow-100.model word2vec

	# # glove
	# invokeps $dpos $dneg data/embs/glove-twitter-100.model word2vec



	# # 300dim 
	# # ----------------------------------
	# # sg
	# invokeps $dpos $dneg data/embs/w2v-twitter-skip-300.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-skip-300.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-skip-300.model word2vec

	# # cbow
	# invokeps $dpos $dneg data/embs/w2v-twitter-cbow-300.model word2vec
	# invokeps $dpos $dneg data/embs/ft-twitter-cbow-300.model word2vec
	# invokeps $dpos $dneg data/embs/wang2v-twitter-cbow-300.model word2vec

	# # glove
	# invokeps $dpos $dneg data/embs/glove-twitter-300.model word2vec


	# 500dim 
	# ----------------------------------
	# sg
	invokeps $dpos $dneg data/embs/w2v-twitter-skip-500.model word2vec
	invokeps $dpos $dneg data/embs/ft-twitter-skip-500.model word2vec

	# cbow
	invokeps $dpos $dneg data/embs/w2v-twitter-cbow-500.model word2vec
	invokeps $dpos $dneg data/embs/ft-twitter-cbow-500.model word2vec

	# glove
	invokeps $dpos $dneg data/embs/glove-twitter-500.model word2vec

	invokeps $dpos $dneg data/embs/wang2v-twitter-cbow-500.model word2vec
	invokeps $dpos $dneg data/embs/wang2v-twitter-skip-500.model word2vec
}


invokeds data/corpora/CLEAN_buscape2.pos data/corpora/CLEAN_buscape2.neg
invokeds data/corpora/CLEAN_buscape1.pos data/corpora/CLEAN_buscape1.neg
invokeds data/corpora/CLEAN_ml.pos data/corpora/CLEAN_ml.neg
invokeds data/corpora/CLEAN_EleicoesPresidenciaisDilma.pos data/corpora/CLEAN_EleicoesPresidenciaisDilma.neg
invokeds data/corpora/CLEAN_EleicoesPresidenciaisSerra.pos data/corpora/CLEAN_EleicoesPresidenciaisSerra.neg



