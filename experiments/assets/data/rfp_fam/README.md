The original RFP dataset was used as listed in LaMBO reference work.        # TODO: CITE
Known sequence structures were extracted and aligned with clustalo binaries (see Clustal Omega v. 1.2.4 64-bit Linux binary) -> MSA.
The aligned MSA was used with HMMER to build a sequence alignment for VAE training using the MPI Bioinformatics Toolkit (see Zimmermann et al.).:
 - Reference hmmsearch against alphafold_uniprot50DB (default).
 - E-value cutoff set to 1 to obtain also distant sequences.
 - columns were filtered gap 20%

(reference Bioinformatics Toolkit Job ID: 3663511 30/08/2023)

CITE:
Lukas Zimmermann, Andrew Stephens, Seung-Zin Nam, David Rau, Jonas Kübler, Marko Lozajic, Felix Gabler, Johannes Söding, Andrei N. Lupas, Vikram Alva,
A Completely Reimplemented MPI Bioinformatics Toolkit with a New HHpred Server at its Core,
Journal of Molecular Biology,
Volume 430, Issue 15,
2018,
Pages 2237-2243,
ISSN 0022-2836,
https://doi.org/10.1016/j.jmb.2017.12.007.
(https://www.sciencedirect.com/science/article/pii/S0022283617305879)
Abstract: The MPI Bioinformatics Toolkit (https://toolkit.tuebingen.mpg.de) is a free, one-stop web service for protein bioinformatic analysis. It currently offers 34 interconnected external and in-house tools, whose functionality covers sequence similarity searching, alignment construction, detection of sequence features, structure prediction, and sequence classification. This breadth has made the Toolkit an important resource for experimental biology and for teaching bioinformatic inquiry. Recently, we replaced the first version of the Toolkit, which was released in 2005 and had served around 2.5 million queries, with an entirely new version, focusing on improved features for the comprehensive analysis of proteins, as well as on promoting teaching. For instance, our popular remote homology detection server, HHpred, now allows pairwise comparison of two sequences or alignments and offers additional profile HMMs for several model organisms and domain databases. Here, we introduce the new version of our Toolkit and its application to the analysis of proteins.
Keywords: MPI Bioinformatics Toolkit; HHpred; HHblits; structure prediction; remote homology detection