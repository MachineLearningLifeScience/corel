# Setup
```
pip install trieste
pip install -e .
```

## Setting up *Poli*
```
git clone poli && cd ./poli/
pip install -e .
```



## Datasets
### RFP
The original RFP dataset was used as listed in LaMBO reference work.
Known sequence structures were extracted and aligned with clustalo binaries (see Clustal Omega v. 1.2.4 64-bit Linux binary) -> MSA.
The aligned MSA was used with HMMER to build a sequence alignment for VAE training using the MPI Bioinformatics Toolkit (see Zimmermann et al.).:
 - Reference hmmsearch against alphafold_uniprot50DB (default).
 - E-value cutoff set to 1 to obtain also distant alignments.
 - columns were filtered gap 20%

See Dataset readme for details and citations.