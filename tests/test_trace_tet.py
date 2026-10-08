"""Focused diagnostic checks without running BWA or refitting models."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ko_variation.trace_tet import main, tet_names


class TetTraceTests(unittest.TestCase):
    def test_gene_names(self):
        self.assertEqual(tet_names('tet(M); Tet(K) efflux'), {'tetM', 'tetK'})
        self.assertEqual(tet_names('TetR-family regulator') & {'tetK', 'tetM'}, set())

    def test_full_trace_retains_multiple_overlaps_and_filter_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root/'bakta'/'S1'
            sample.mkdir(parents=True)
            (sample/'S1.fna').write_text('>contig_1\n'+'A'*2500+'\n')
            (sample/'S1.gff3').write_text(
                'contig_1\tBakta\tCDS\t500\t800\t.\t+\t0\tID=a;gene=tet(K);product=Tet(K) efflux\n'
                'contig_1\tBakta\tCDS\t520\t600\t.\t+\t0\tID=b;product=hypothetical protein\n'
                'contig_1\tBakta\tCDS\t1500\t1800\t.\t-\t0\tID=c;gene=tet(M);product=Tet(M) protection\n')
            (root/'binary.fa').write_text('>S1\nCACCC\n')
            (root/'unitigs').write_text('0 '+'A'*61+'\n1 '+'C'*61+'\n2 '+'G'*61+'\n')
            records = [
                dict(u=0,v=2,status='LOW_CELL_COUNT',p_primary=None,min_distance=20000,n11=1,n10=0,n01=0,n00=1),
                dict(u=1,v=2,status='OK',p_primary=1e-10,min_distance=20000,n11=1,n10=1,n01=1,n00=1),
                dict(u=1,v=3,status='OK',p_primary=1e-9,min_distance=100,n11=1,n10=1,n01=1,n00=1),
                dict(u=0,v=4,status='OK',p_primary=.5,min_distance=20000,n11=1,n10=1,n01=1,n00=1)]
            pd.DataFrame(records).to_csv(root/'results.tsv',sep='\t',index=False)
            (root/'pairs').write_text('0 2 20000\n1 2 20000\n1 3 100\n0 4 20000\n')
            output = root/'trace'
            argv = ['--unitigs',str(root/'unitigs'),'--fasta',str(root/'binary.fa'),
                    '--bakta',str(root/'bakta'),'--results',str(root/'results.tsv'),
                    '--out',str(output),'--pairs',str(root/'pairs'),'--chunk-rows','2','--resume']
            def fake_bwa(command, **kwargs):
                if command[1] == 'fastmap':
                    # Extracted windows merge and begin at original coordinate 1.
                    kwargs['stdout'].write('SQ\t0\t61\nEM\t0\t61\t1\ttetregion0:+520\n//\n'
                                          'SQ\t1\t61\nEM\t0\t61\t1\ttetregion0:-1801\n//\n'
                                          'SQ\t2\t61\nEM\t0\t61\t10001\t*\n//\n')
            with patch('ko_variation.trace_tet.shutil.which', return_value='bwa'), patch('ko_variation.trace_tet.subprocess.run', side_effect=fake_bwa) as runner:
                self.assertEqual(main(argv),0)
                self.assertEqual(runner.call_count,2)
                self.assertEqual(main(argv),0)
                self.assertEqual(runner.call_count,2)  # mapping reused
            summary = json.loads((output/'trace_summary.json').read_text())
            self.assertEqual(summary['target_pair_stage_counts'],
                {'ineligible_or_failed':1,'selected_distal':1,'excluded_distance':1,'not_significant':1})
            self.assertEqual(summary['mapping']['repeat_limited_unitigs'],1)
            loci = pd.read_csv(output/'tet_locus_trace.tsv',sep='\t').set_index('locus')
            self.assertEqual(loci.loc[0,'target_genes'],'tetK')
            self.assertEqual(loci.loc[0,'n_hits_overlapping_multiple_cds'],1)
            self.assertEqual(loci.loc[1,'annotation_classes'],'upstream')
            self.assertEqual(loci.loc[1,'n_exact_hit_samples_binary_absent'],1)
            self.assertTrue((root/'results.tsv').is_file())


if __name__ == '__main__':
    unittest.main()
