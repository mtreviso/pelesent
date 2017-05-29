#!/bin/bash
# rodar: sudo ./run.sh

# fixed
embf="data/embs/w2v-twitter-skip-300.model"
embt="word2vec"
dpath="data/corpora"

# var
emoticonsPOS="${dpath}/CLEAN_filtro_emoticons+emojis.pos"
emoticonsNEG="${dpath}/CLEAN_filtro_emoticons+emojis.neg"

dilmaPOS="${dpath}/CLEAN_EleicoesPresidenciaisDilma.pos"
dilmaNEG="${dpath}/CLEAN_EleicoesPresidenciaisDilma.neg"

serraPOSOK="${dpath}/CLEAN_EleicoesPresidenciaisSerra.pos"
serraNEGOK="${dpath}/CLEAN_EleicoesPresidenciaisSerra.neg"

serraPOS="${dpath}/CLEAN_FIXED_EleicoesPresidenciaisSerra.pos"
serraNEG="${dpath}/CLEAN_FIXED_EleicoesPresidenciaisSerra.neg"


buscape1POS="${dpath}/CLEAN_buscape1.pos"
buscape1NEG="${dpath}/CLEAN_buscape1.neg"

buscape2POS="${dpath}/CLEAN_buscape2.pos"
buscape2NEG="${dpath}/CLEAN_buscape2.neg"

mlPOS="${dpath}/CLEAN_ml.pos"
mlNEG="${dpath}/CLEAN_ml.neg"


invokeps(){
	local pf=$1
	local nf=$2
	local tpf=$3
	local tnf=$4
	local bt=$5
	time sudo python3 -m pelesent --gpu --pos-file $pf --neg-file $nf --test-pos-file $tpf --test-neg-file $tnf --emb-file $embf --emb-type $embt --batch-size $bt
}


# invokeps $emoticonsPOS $emoticonsNEG $buscape2POS $buscape2NEG 
# invokeps $emoticonsPOS $emoticonsNEG $buscape1POS $buscape1NEG 
# invokeps $emoticonsPOS $emoticonsNEG $mlPOS $mlNEG
# invokeps $emoticonsPOS $emoticonsNEG $dilmaPOS $dilmaNEG
invokeps $emoticonsPOS $emoticonsNEG $serraPOS $serraNEG 128
invokeps $emoticonsPOS $emoticonsNEG $serraPOS $serraNEG 32
invokeps $emoticonsPOS $emoticonsNEG $serraPOSOK $serraNEGOK 32