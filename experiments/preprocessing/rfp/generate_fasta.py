__author__ = 'Simon Bartels'

from Bio.SeqIO import SeqRecord, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio import SeqIO
import csv


WRITE = True

ONLY_KNOWN_STRUCTURES = False

SMALL = True


seqs = []
if ONLY_KNOWN_STRUCTURES:
    f = open('./lambo/assets/fpbase/rfp_known_structures.csv', newline='')
else:
    f = open('./lambo/assets/fpbase/fpbase_sequences.csv', newline='')
#f = './lambo/assets/fpbase/fpbase_sequences.csv'
for row in csv.reader(f):
    if ONLY_KNOWN_STRUCTURES:
        s = SeqRecord(Seq(row[14]), id=row[0], name=row[0])
    else:
        s = SeqRecord(Seq(row[4]), id=row[1], name=row[1])
    #if len(s) > 450:
    #    print("wtf?!")
    if SMALL and len(s) > 250:  # That is the length that LaMBO excludes
        continue
    seqs.append(s)

if ONLY_KNOWN_STRUCTURES:
    assert(seqs.pop(0).seq == 'fpbase_seq')
else:
    assert(seqs.pop(0).seq == 'Seq')

if ONLY_KNOWN_STRUCTURES:
    name = 'seqs_known_structures.fasta'
else:
    name = 'seqs.fasta'
if SMALL:
    name = 'small_' + name

if WRITE:
    f = open(name, 'w+')
    SeqIO.write(seqs, f, 'fasta')
    f.close()

# if ONLY_KNOWN_STRUCTURES:
#     f = open('seqs_known_structures.fasta', 'r')
# else:
#     f = open('seqs.fasta', 'r')
f = open(name, 'r')

seqs_ = list(SeqIO.parse(f, 'fasta'))
print(len(seqs_))
assert(len(seqs_) == len(seqs))

i = 0
max_len = 0
for s, s_ in zip(seqs, seqs_):
    assert(s.seq == s_.seq, "%i" % i)
    if len(s_.seq) > max_len:
        max_len = len(s_.seq)
    i += 1
print(max_len)
