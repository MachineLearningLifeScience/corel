#!/bin/bash
rm -r output/*
mkdir output
hmmer=../../../hmmer/bin
seqs_file=data/small_seqs.fasta
${hmmer}/hmmbuild output/rfp.hmm data/wildtype_mrouge.fasta
#${hmmer}/hmmbuild -O output/rfp.stk output/rfp.hmm data/wildtype_mrouge.fasta
#${hmmer}/hmmalign -o output/rfp.stk output/rfp.hmm ${seqs_file}
#${hmmer}/hmmbuild -O output/rfp.stk output/rfp.hmm output/rfp.stk
#${hmmer}/hmmalign --outformat A2M -o output/rfp.a2m output/rfp.hmm ${seqs_file}
#${hmmer}/hmmbuild output/rfp.hmm output/rfp.a2m
${hmmer}/hmmalign --trim -o output/wt.stk output/rfp.hmm data/wildtype_mrouge.fasta
${hmmer}/hmmalign --trim -o output/rfp.stk output/rfp.hmm ${seqs_file}
${hmmer}/hmmbuild output/rfp.hmm output/rfp.stk
#${hmmer}/hmmalign -o output/rfp_new.stk --trim --mapali output/rfp.stk output/rfp.hmm output/rfp.stk
${hmmer}/hmmalign -o output/rfp_new.stk output/rfp.hmm ${seqs_file}
${hmmer}/hmmalign -o output/rfp_new.a2m --outformat clustal output/rfp.hmm ${seqs_file}
